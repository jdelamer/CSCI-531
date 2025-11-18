\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{csbeamer}[202/01/03 CS Beamer class]

\DeclareOption{handout}{\PassOptionsToClass{\CurrentOption}{beamer}} 
\ProcessOptions\relax
\LoadClass[aspectratio=169]{beamer}



% ======================================
% ============== Packages ==============
% ======================================

% ============== Format ================
\RequirePackage{caption}
\RequirePackage[T1]{fontenc}

% ============== Format ================
\RequirePackage{xcolor}

% ============== Graphic ===============
\RequirePackage{graphicx}


\RequirePackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning}

\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
% ======================================
% ==============  Themes  ==============
% ======================================

% =========== Informations =============


% \logo{
% \includegraphics[width=3cm]{LogoWordsBottom.png}
% }

\titlegraphic{
    \includegraphics[width=4cm]{LogoWordsBottom.png}
}

\newcommand{\setuniversity}{St. Francis Xavier University}
\newcommand{\university}[1]{\renewcommand{\setuniversity}{#1}}

\newcommand{\setdepartment}{Department of Computer Science}
\newcommand{\department}[1]{\renewcommand{\setdepartment}{#1}}

\newcommand{\setcourse}{}
\newcommand{\course}[1]{\renewcommand{\setcourse}{#1}}

\newcommand{\setcourseshort}{}
\newcommand{\courseshort}[1]{\renewcommand{\setcourseshort}{#1}}

\newcommand{\setterm}{}
\newcommand{\term}[1]{\renewcommand{\setterm}{#1}}

% ============= Colors ===============

\definecolor{stfxblue}{HTML}{001b40}
\definecolor{skyblue}{HTML}{425f8e}
\definecolor{skylight}{HTML}{5e739e}
\definecolor{marigold}{HTML}{d5a845}

\setbeamercolor{frametitle}{fg=stfxblue}
\setbeamercolor{title}{fg=stfxblue}


\setbeamercolor{head}{use=structure,fg=white,bg=stfxblue}
\setbeamercolor{foot left}{use=structure,fg=white,bg=stfxblue}
\setbeamercolor{foot middle}{use=structure,fg=white,bg=skylight}
\setbeamercolor{foot right}{fg=white,bg=skyblue}

%\setbeamertemplate{background}{
%    \begin{center}
%        \includegraphics[width=\paperwidth]{LogoWordsBottom.png}
%    \end{center}
%}

% ============= Headline ===============

\setbeamertemplate{headline}{
    \leavevmode%
    \hbox{%
        \begin{beamercolorbox}[wd=\paperwidth,ht=2.75ex,dp=1ex,center]{head}%
            \usebeamerfont{section in head/foot}\insertshorttitle\hspace*{2ex}
        \end{beamercolorbox}
    }
}

% ============= Footline ===============

\setbeamertemplate{footline}{
    \leavevmode%
    \hbox{%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{foot left}%
            \usebeamerfont{title in head/foot}\setcourseshort
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{foot middle}%
            \usebeamerfont{title in head/foot}\insertauthor
        \end{beamercolorbox}%
        \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{foot right}%
            \usebeamerfont{date in head/foot}\setterm{}\hfill
            \insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
        \end{beamercolorbox}
    }%
\vskip0pt%
}

% ============== Title ================

\setbeamertemplate{title page}{
  \vbox{}
  \vfill
  \begingroup
    \centering
    \usebeamercolor[fg]{title}\Huge\inserttitle
    \vskip1em
    \inserttitlegraphic
    \vskip1em
    \large\setcourse
    \vskip1em
    \setterm
    \vskip1em
  \endgroup
  \vfill
}

% =========== Frame Title =============

\setbeamertemplate{frametitle}{
    \begin{center}
        \insertframetitle
    \end{center}
}
