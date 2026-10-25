"""Strict point SCR broker: exact native/source identities, original lifecycle.

Independent corrected module derived from the historical 2026-10-16 broker
whose SHA-256 is 609302b76decc2c90011869582c4180b693a40cc7e3267f77abca39312692c73.
Historical experiment sources and results are intentionally unchanged.

Canonical source rows use signed SQLite int64 IDs (negative IDs are valid),
positive int64 version IDs, nonnegative integer commits and 0/1 tombstones.
Native keys must be Arrow integers with exactly those domains. No key coercion
is permitted. One fixed encoder and sealed, complete source history remain
input contracts, as in the historical local authority. Rejected publication
rolls back registry entries and any ownership lock newly acquired by that call.
"""
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
    @staticmethod
    def _exact_key(row,version):
        if (not isinstance(row,int) or isinstance(row,bool) or not -(1<<63)<=row<(1<<63)
            or not isinstance(version,int) or isinstance(version,bool) or not 1<=version<(1<<63)):
            raise ValueError('keys require exact SQLite int64 row and positive int64 version')
        return row,version
    @staticmethod
    def _native_key_schema(arrow):
        import pyarrow as pa
        for name in ('row','version'):
            if arrow.column_names.count(name)!=1 or not pa.types.is_integer(arrow.schema.field(name).type):
                raise ValueError('native key fields must have Arrow integer types')
    @staticmethod
    def _validated_frontier(records):
        if len(records)!=1:raise ValueError('source requires exactly one frontier record')
        head,floor=records[0]
        if any(not isinstance(x,int) or isinstance(x,bool) for x in (head,floor)) or not 0<=floor<=head:
            raise ValueError('source frontier requires nonnegative integer head/floor')
        return head,floor
    def _read_source_state(self,connection,s=None):
        """Caller holds a read or reserved write transaction; no source writes."""
        head,floor=self._validated_frontier(connection.execute('select epoch,floor from meta').fetchall())
        if s is None:s=head
        if not isinstance(s,int) or isinstance(s,bool):raise ValueError('integer commit cut required')
        if not floor<=s<=head:raise ValueError('snapshot outside committed readable frontier')
        records=connection.execute('select row,epoch,version,deleted,vector,payload from history order by row,epoch').fetchall()
        keys,previous,state=set(),{},{}
        for r,epoch,v,deleted,vec,payload in records:
            key=self._exact_key(r,v)
            if (not isinstance(epoch,int) or isinstance(epoch,bool) or not 0<=epoch<=head
                or not isinstance(deleted,int) or isinstance(deleted,bool) or deleted not in (0,1)):
                raise ValueError('source history requires committed integer epochs and 0/1 tombstones')
            if key in keys or previous.get(r)==epoch:
                raise ValueError('source version identity reused or ambiguous row commit')
            keys.add(key);previous[r]=epoch
            if not deleted and (not isinstance(vec,(bytes,bytearray,memoryview)) or not len(vec) or len(vec)%4):
                raise ValueError('source embedding requires float32 bytes')
            if epoch<=s:state[r]=(v,deleted,bytes(vec) if not deleted else b'',payload)
        return head,floor,{r:(v,vec,payload) for r,(v,deleted,vec,payload) in state.items() if not deleted}
    def _configured_table(self,table):
        configured=self.native_db.open_table(table.name)
        supplied,actual=os.stat(table.uri),os.stat(configured.uri)
        if (supplied.st_dev,supplied.st_ino)!=(actual.st_dev,actual.st_ino):
            raise ValueError('native table outside configured artifact domain')
        return configured
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
        table=self._configured_table(table)
        identity=os.stat(table.uri);key=(identity.st_dev,identity.st_ino)
        if key not in self._native_owners:
            self._native_owners[key]=_retention_owner(identity,'native table','native-')
        return key
    def frontier(self,s):
        if self._closed:raise ValueError('retention authority closed')
        if not isinstance(s,int) or isinstance(s,bool):raise ValueError('integer commit cut required')
        if self.reader is self.sql and self.sql.in_transaction:raise ValueError('uncommitted source transaction')
        head,floor=self._validated_frontier(self.reader.execute('select epoch,floor from meta').fetchall())
        if not floor<=s<=head:raise ValueError('snapshot outside committed readable frontier')
        return head,floor
    def source(self,s):
        with self.lock:
            if self._closed:raise ValueError('retention authority closed')
            if not isinstance(s,int) or isinstance(s,bool):raise ValueError('integer commit cut required')
            self.reader.execute('begin')
            try:
                return self._read_source_state(self.reader,s)[2]
            finally:self.reader.rollback()
    def publish(self,name,s,table):
        with self.lock:
            if name in self.manifests:raise ValueError('duplicate manifest')
            v=table.version() if callable(table.version) else table.version
            if not isinstance(v,int) or isinstance(v,bool) or v<1:raise ValueError('native version identity must be a positive integer')
            private=self._configured_table(table)
            identity=os.stat(private.uri);domain=(identity.st_dev,identity.st_ino)
            already_owned=domain in self._native_owners
            self._own_table(private)
            try:
                private.checkout(v)
                source=self.source(s);actual=private.to_arrow();population={}
                import numpy as np,pyarrow as pa
                self._native_key_schema(actual)
                if actual.column_names.count('vector')!=1:raise ValueError('one native vector field required')
                vector_type=actual.schema.field('vector').type
                if not pa.types.is_fixed_size_list(vector_type) or not pa.types.is_float32(vector_type.value_type):
                    raise ValueError('native embeddings must be fixed-size Float32 vectors')
                for record in actual.to_pylist():
                    key=self._exact_key(record['row'],record['version'])
                    vector=np.asarray(record['vector'],dtype=np.float32)
                    if key in population or not np.isfinite(vector).all():raise ValueError('duplicate candidate or nonfinite embedding')
                    population[key]=vector.tobytes()
                for row,(version,vector,payload) in source.items():
                    if population.get((row,version))!=vector:raise ValueError('snapshot population or embedding mismatch')
                view=MappingProxyType({row:(value[0],value[2]) for row,value in source.items()})
                manifest=NativeManifest(name,s,s,table.name,v,frozenset(population),view)
                self.manifests[name]=manifest;self.support[name]=(s,s);self._manifest_domains[name]=domain;return manifest
            except BaseException:
                self.manifests.pop(name,None);self.support.pop(name,None);self._manifest_domains.pop(name,None)
                if not already_owned:self._native_owners.pop(domain).close()
                raise
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
            result=handle.search(vector).nprobes(m.settings[0]).ef(m.settings[1]).limit(40).to_arrow()
            self._native_key_schema(result);records=result.to_pylist()
            actual=handle.version() if callable(handle.version) else handle.version
            if actual!=m.native_version:raise ValueError('native generation identity changed during search')
            answers=[]
            for r in records:
                key=self._exact_key(r['row'],r['version'])
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
            if self._closed:raise ValueError('retention authority closed')
            if self.sql.in_transaction:raise ValueError('source write must commit before reclamation')
            protected=[s for s,n in self.pins.values()]+[v[0] for v in self.support.values() if v is not None]
            # The collector never executes on the caller's writer connection.
            # Its reserved transaction seals the head/floor read and deletion.
            self.collector.execute('begin immediate')
            try:
                # Validate the same canonical frontier and retained lineage used
                # for certification before any DELETE or frontier UPDATE.
                head,oldfloor,_=self._read_source_state(self.collector)
                if any(not isinstance(s,int) or isinstance(s,bool) or not oldfloor<=s<=head for s in protected):
                    raise ValueError('protected snapshot outside committed readable frontier')
                floor=min(protected) if protected else head
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
