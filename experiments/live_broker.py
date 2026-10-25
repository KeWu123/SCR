"""Native SCR reference broker: SQLite version history and LanceDB artifacts."""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
import json,sqlite3,datetime,os,tempfile
from pathlib import Path

def _retention_owner(identity,label,namespace=''):
    """Cross-process ownership follows physical identity, including path aliases."""
    ownerdir=Path(tempfile.gettempdir())/'scr-retention-locks';ownerdir.mkdir(exist_ok=True)
    owner=open(ownerdir/f'{namespace}{identity.st_dev}-{identity.st_ino}.lock','a+b')
    owner.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(owner.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        owner.close();raise ValueError(f'{label} retention authority already active') from None
    return owner

@dataclass(frozen=True)
class NativeManifest:
    name:str
    low:int
    high:int
    table_name:str
    native_version:int
    keys:frozenset
    source_view:object
    settings:tuple=(4,128,10)
class NativeBroker:
    def __init__(self,sql,native_db):
        self.sql=sql;self.native_db=native_db;self.lock=RLock();self.manifests={};self.support={};self.pins={};self.ops={};self.serial=0;self._ending=set()
        if 'floor' not in {r[1] for r in sql.execute('pragma table_info(meta)')}:raise ValueError('source must expose a durable readable frontier')
        filename=sql.execute('pragma database_list').fetchone()[2]
        if not filename:raise ValueError('native broker requires a persisted source domain')
        # One retention authority per source, including across processes. Source
        # writers may commit through independent connections; only this authority
        # prunes history or deletes protected native versions.
        self._owner=_retention_owner(os.stat(filename),'source')
        self._native_owners={};self._manifest_domains={}
        self.reader=sqlite3.connect(filename,check_same_thread=False,isolation_level=None) if filename else sql
        if filename:self.reader.execute('pragma query_only=ON')
        self.collector=sqlite3.connect(filename,check_same_thread=False,isolation_level=None,timeout=30)
        self._closed=False
    def close(self):
        with self.lock:
            if self._closed:return
            if self.pins or self.ops:raise ValueError('release active references before closing authority')
            self.support.clear();self.manifests.clear();self._manifest_domains.clear()
            self.reader.close();self.collector.close();self._owner.close();self._closed=True
            for owner in self._native_owners.values():owner.close()
            self._native_owners.clear()
    def _own_table(self,table):
        """Caller holds self.lock. Local native directories share one authority.

        Source writers may append native generations and maintain conservatively.
        Destructive native reclamation must use cleanup on the owning authority.
        Ownership lasts until close, including after the last manifest is retired.
        """
        if self._closed:raise ValueError('retention authority closed')
        identity=os.stat(table.uri);key=(identity.st_dev,identity.st_ino)
        if key not in self._native_owners:
            self._native_owners[key]=_retention_owner(identity,'native table','native-')
        return key
    def frontier(self,s):
        if self._closed:raise ValueError('retention authority closed')
        if not isinstance(s,int) or isinstance(s,bool):raise ValueError('integer commit cut required')
        if self.reader is self.sql and self.sql.in_transaction:raise ValueError('uncommitted source transaction')
        head,floor=self.reader.execute('select epoch,floor from meta').fetchone()
        if not floor<=s<=head:raise ValueError('snapshot outside committed readable frontier')
        return head,floor
    def source(self,s):
        with self.lock:
            self.reader.execute('begin')
            try:
                self.frontier(s)
                records=self.reader.execute('select row,version,deleted,vector,payload from history h where epoch=(select max(epoch) from history z where z.row=h.row and z.epoch<=?) and h.epoch<=? order by row',(s,s)).fetchall()
                return {r:(v,bytes(vec),payload) for r,v,deleted,vec,payload in records if not deleted}
            finally:self.reader.rollback()
    def publish(self,name,s,table):
        with self.lock:
            if name in self.manifests:raise ValueError('duplicate manifest')
            v=table.version() if callable(table.version) else table.version
            private=self.native_db.open_table(table.name)
            supplied,actual_table=os.stat(table.uri),os.stat(private.uri)
            if (supplied.st_dev,supplied.st_ino)!=(actual_table.st_dev,actual_table.st_ino):raise ValueError('native table outside configured artifact domain')
            domain=self._own_table(private);private.checkout(int(v))
            source=self.source(s);actual=private.to_arrow().to_pylist();population={}
            import numpy as np
            for record in actual:
                key=(int(record['row']),int(record['version']));value=np.array(record['vector'],dtype=np.float32).tobytes()
                if key in population:raise ValueError('duplicate native candidate')
                population[key]=value
            for row,(version,vector,payload) in source.items():
                if population.get((row,version))!=vector:raise ValueError('snapshot population or embedding mismatch')
            view=MappingProxyType({row:(value[0],value[2]) for row,value in source.items()})
            manifest=NativeManifest(name,s,s,table.name,int(v),frozenset(population),view)
            self.manifests[name]=manifest;self.support[name]=(s,s);self._manifest_domains[name]=domain;return manifest
    def checkout(self,manifest):
        """An independent handle; neither descriptors nor calls expose aliases."""
        handle=self.native_db.open_table(manifest.table_name)
        handle.checkout(manifest.native_version)
        return handle
    def begin(self,s,name,hook=None):
        with self.lock:
            support=self.support.get(name)
            if support is None or not support[0]<=s<=support[1]:raise ValueError('snapshot support closed')
            m=self.manifests[name]
            self.frontier(s)
            handle=self.checkout(m)
            actual=handle.version() if callable(handle.version) else handle.version
            if actual!=m.native_version:raise ValueError('native generation identity changed')
            if s!=m.low or s!=m.high:raise ValueError('reference broker supports exact snapshot manifests')
            self.serial+=1;token=self.serial;self.pins[token]=(s,name)
            try:
                if hook:hook()
            except BaseException:del self.pins[token];raise
            return token
    def query(self,token,vector,hook=None):
        with self.lock:
            if token in self._ending:raise ValueError('episode closing')
            s,name=self.pins[token];m=self.manifests[name];source=m.source_view;self.ops[token]=self.ops.get(token,0)+1
        try:
            handle=self.checkout(m)
            actual=handle.version() if callable(handle.version) else handle.version
            if actual!=m.native_version:raise ValueError('native generation identity changed')
            if hook:hook()
            records=handle.search(vector).nprobes(m.settings[0]).ef(m.settings[1]).limit(40).to_arrow().to_pylist()
            actual=handle.version() if callable(handle.version) else handle.version
            if actual!=m.native_version:raise ValueError('native generation identity changed during search')
            answers=[]
            for r in records:
                key=(int(r['row']),int(r['version']))
                if key[0] in source and source[key[0]][0]==key[1]:answers.append((key[0],key[1],source[key[0]][1]))
            return answers[:m.settings[2]]
        finally:
            with self.lock:
                self.ops[token]-=1
                if not self.ops[token]:del self.ops[token]
                if token in self._ending and token not in self.ops:del self.pins[token];self._ending.remove(token)
    def audit_hold(self,token):
        with self.lock:
            if token in self._ending:raise ValueError('episode closing')
            self.serial+=1;self.pins[self.serial]=self.pins[token];return self.serial
    def withdraw(self,name):
        with self.lock:self.support[name]=None
    def end(self,token):
        with self.lock:
            if token in self.ops:self._ending.add(token)
            else:del self.pins[token]
    def gc(self):
        with self.lock:
            if self.sql.in_transaction:raise ValueError('source write must commit before reclamation')
            protected=[s for s,n in self.pins.values()]+[v[0] for v in self.support.values() if v is not None]
            # The collector never executes on the caller's writer connection.
            # Its reserved transaction seals the head/floor read and deletion.
            self.collector.execute('begin immediate')
            try:
                head,oldfloor=self.collector.execute('select epoch,floor from meta').fetchone();floor=min(protected) if protected else head
                if floor<oldfloor:raise ValueError('readable frontier cannot move backwards')
                self.collector.execute('delete from history where epoch<? and epoch<(select max(epoch) from history z where z.row=history.row and z.epoch<=?)',(floor,floor))
                self.collector.execute('update meta set floor=? where floor<=?',(floor,floor))
                if self.collector.execute('select changes()').fetchone()[0]!=1:raise ValueError('invalid source frontier state')
                self.collector.commit()
            except BaseException:self.collector.rollback();raise
            removed=[]
            for name in list(self.manifests):
                if self.support[name] is None and not any(n==name for s,n in self.pins.values()):removed.append(self.manifests.pop(name));del self.support[name];del self._manifest_domains[name]
            return floor,removed
    def maintain(self,table):
        with self.lock:
            self._own_table(table)
            oldest=min(v['timestamp'] for v in table.list_versions())
            keep=datetime.datetime.now()-oldest+datetime.timedelta(days=1)
            table.optimize(cleanup_older_than=max(keep,datetime.timedelta(days=1)))
    def cleanup(self,table):
        """Reclaim native versions only after all affected support and pins close."""
        with self.lock:
            domain=self._own_table(table)
            affected={name for name,key in self._manifest_domains.items() if key==domain}
            if any(self.support[name] is not None for name in affected) or any(name in affected for s,name in self.pins.values()):
                raise ValueError('close native table support and active references before cleanup')
            return table.optimize(cleanup_older_than=datetime.timedelta(seconds=0),delete_unverified=True)
