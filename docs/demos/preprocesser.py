<%
def mod(x):
    return '`'+x+'`'
def cls(x):
    return '`'+x+'`'
def func(x):
    return '`'+x+'`'
%>
% if FORMAT in ("sphinx"):
<%
def mod(x):
    if '.' in x:
        return ':mod:'+'`'+x+'`'
    return ':mod:'+'`'+'.'+x+'`'
def cls(x):
    if '.' in x:
        return ':class:'+'`'+x+'`'
    return ':class:'+'`'+'.'+x+'`'
def func(x):
    if '.' in x:
        return ':func:'+'`'+x+'`'
    return ':func:'+'`'+'.'+x+'`'
%>
% elif FORMAT in ("ipynb"):
<%

def mod(xs):
    path = r'https://shenfun.readthedocs.io/en/latest'
    xp = xs.split('.')
    fl = ".".join(xp[:-1]) if xs != 'shenfun' else xs
    s = __import__(xs, globals(), locals(), [xp[-1]])
    #ss = "<a class='reference internal' href='%s/%s.html#module-%s' title=%s><code class='xref py py-mod docutils literal notranslate'><span class='pre'>%s</span></code></a>"%(path, fl, xs, xs, xs)
    ss = "[%s](%s/%s.html#module-%s)"%(xs, path, fl, xs)
    return ss

def cls(xs):
    import shenfun
    try:
        x = vars(shenfun)[xs]
    except KeyError:
        xss = xs.split('.')
        assert len(xss) == 2
        x0 = vars(shenfun)[xss[0]]
        x = vars(x0)[xss[1]]
        xs = xss[1]
    m = x.__module__
    xp = m.split('.')
    fl = ".".join(xp[:-1])
    sl = ".".join(xp+[xs])
    path = r'https://shenfun.readthedocs.io/en/latest'
    #ss = "<a class='reference internal' href='%s/%s.html#%s.%s' title='%s.%s'><code class='xref py py-func docutils literal notranslate'><span class='pre'>%s</span></code></a>"%(path, fl, m, xs, m, xs, xs)
    ss = "[%s](%s/%s.html#%s.%s)"%(xs, path, fl, m, xs)
    return ss

def func(xs):
    import shenfun
    import inspect
    x = vars(shenfun)[xs]
    m = x.__module__
    xp = m.split('.')
    if inspect.getfile(x).split('/')[-1] == '__init__.py':
        fl = m
    else:
        fl = ".".join(xp[:-1])
    sl = ".".join(xp+[xs])
    path = r'https://shenfun.readthedocs.io/en/latest'
    #ss = "<a class='reference internal' href='%s/%s.html#%s.%s' title='%s.%s'><code class='xref py py-func docutils literal notranslate'><span class='pre'>%s()</span></code></a>"%(path, fl, m, xs, m, xs, xs)
    ss = "[%s()](%s/%s.html#%s.%s)"%(xs, path, fl, m, xs)
    return ss
%>
% endif

