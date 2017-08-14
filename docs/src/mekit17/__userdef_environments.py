import re

def latex_minipage(text, titleline, counter, format):
    # file=myprog.py label=my:label Some title...
    label, titleline = get_label(titleline, 'label')
    filename, titleline = get_label(titleline, 'file')
    # Must be able to handle empty label and/or filename
    # (recognized by '')
    if label:
        label = 'label{%s}' % label
    if filename:
        filename = '`%s`: ' % filename
    s = r"""

\vskip 1ex
\noindent
\begin{minipage}{\columnwidth}

%s


\end{minipage}
""" % (text)
    return s

def latex_code(text, titleline, counter, format):
    # file=myprog.py label=my:label Some title...
    label, titleline = get_label(titleline, 'label')
    filename, titleline = get_label(titleline, 'file')
    # Must be able to handle empty label and/or filename
    # (recognized by '')
    if label:
        label = 'label{%s}' % label
    if filename:
        filename = '`%s`: ' % filename
    s = r"""

\begin{pycode}
%s
%s
\hrule
\vspace{1mm}
\hrule

%s

\hrule
\vspace{1mm}
\hrule
\end{pycode}
""" % (label, titleline, text)
    return s

def do_code(text, titleline, counter, format):
    # file=myprog.py label=my:label Some title...
    label, titleline = get_label(titleline, 'label')
    filename, titleline = get_label(titleline, 'file')
    s = r"""

_Python code %d_: label{%s}

%s
""" % (counter, label, text)
    return s

def get_label(titleline, label_text='label'):
    label = ''
    if label_text in titleline:
        pattern = r'%s=([^\s]+)' % label_text
        m = re.search(pattern, titleline)
        if m:
            label = m.group(1)
            titleline = re.sub(pattern, '', titleline).strip()
    return label, titleline

envir2format = {
    'intro': {
        'latex': r"""
\usepackage{amsthm}
\theoremstyle{definition}
\newtheorem{pycode}{Python code}[section]
""",},
    'minipage': {
        'latex': latex_minipage,
        },
    'code': {
        'latex': latex_code,
        'do': do_code
        }
}
