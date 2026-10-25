"""Finite executable SCR manifest specification; immutable offline source ledger."""
from dataclasses import dataclass
from threading import RLock
@dataclass(frozen=True)
class Version:
    row:str
    seq:int
    key:str
    embedding:tuple|None
    payload:str
@dataclass(frozen=True)
class Manifest:
    name:str
    low:int
    high:int
    horizon:int
    receipts:tuple
    population:tuple
    settings:tuple=('l2',10,64)
class Registry:
    def __init__(self,source):
        self.source=tuple(source)
        if [v.seq for v in self.source]!=list(range(1,len(self.source)+1)):raise ValueError('source prefix gap')
        if len({v.key for v in self.source})!=len(self.source):raise ValueError('reused version identity')
        self.rows={v.key:v for v in self.source};self.manifests={};self.open_support={};self.pins={};self.counter=0;self.lock=RLock()
    def certify(self,m,interval=None):
        low,high=interval or (m.low,m.high)
        if not 0<=low<=high<=m.horizon<=len(self.source):raise ValueError('unsupported source range')
        if tuple(m.receipts)!=tuple(range(1,m.horizon+1)):raise ValueError('ingestion prefix gap')
        population=dict(m.population)
        if len(population)!=len(m.population):raise ValueError('duplicate candidate')
        expected={v.key:v for v in self.source}
        for key,embedding in population.items():
            if key not in expected or expected[key].seq>m.horizon or expected[key].embedding is None or expected[key].embedding!=embedding:raise ValueError('candidate identity mismatch')
        for j,v in enumerate(self.source):
            end=next((later.seq for later in self.source[j+1:] if later.row==v.row),len(self.source)+1)
            if v.seq<=high and end>low:
                if self.rows.get(v.key)!=v:raise ValueError('unreadable source history')
                if v.embedding is not None and population.get(v.key)!=v.embedding:raise ValueError('missing historical embedding')
        return True
    def publish(self,m):
        with self.lock:
            if m.name in self.manifests:raise ValueError('manifest already exists')
            frozen=Manifest(m.name,m.low,m.high,m.horizon,tuple(m.receipts),tuple((key,tuple(embedding)) for key,embedding in m.population),tuple(m.settings))
            self.certify(frozen);self.manifests[m.name]=frozen;self.open_support[m.name]=(m.low,m.high)
    def begin(self,s,name,selected_hook=None):
        with self.lock:
            support=self.open_support.get(name)
            if support is None or not support[0]<=s<=support[1]:raise ValueError('snapshot admission closed')
            m=self.manifests[name];self.certify(m,(s,s))
            self.counter+=1;token=self.counter;self.pins[token]=(s,name)
            try:
                if selected_hook:selected_hook()
            except BaseException:del self.pins[token];raise
            return token
    def resolve(self,token):
        with self.lock:
            s,name=self.pins[token];selected={}
            for v in sorted(self.rows.values(),key=lambda v:v.seq):
                if v.seq<=s:selected[v.row]=v
            return {row:v for row,v in selected.items() if v.embedding is not None}
    def withdraw(self,name):
        with self.lock:self.open_support[name]=None
    def end(self,token):
        with self.lock:del self.pins[token]
    def gc(self):
        with self.lock:
            protected=[s for s,n in self.pins.values()]+[v[0] for v in self.open_support.values() if v is not None]
            floor=min(protected) if protected else len(self.source)
            anchors={}
            for v in self.source:
                if v.seq<=floor:anchors[v.row]=v.key
            keep=set(anchors.values())|{v.key for v in self.source if v.seq>floor}
            self.rows={k:v for k,v in self.rows.items() if k in keep}
            removed=[]
            for name in list(self.manifests):
                if self.open_support[name] is None and not any(n==name for s,n in self.pins.values()):removed.append(name);del self.manifests[name];del self.open_support[name]
            return floor,removed
