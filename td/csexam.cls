\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{csexam}[2024/12/16 CS Exam class]

\LoadClass[addpoints]{exam}

% ======================================
% ============== Packages ==============
% ======================================

% ============ Algorithms ==============
\RequirePackage{algorithm}
\RequirePackage[noend]{algpseudocode}

% =============== Math =================
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}

% ============== Format ================
\RequirePackage{caption}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{parskip}

% ============== Graphic ===============
\RequirePackage{graphicx}

% ================ Code ================
%\RequirePackage{minted}

\RequirePackage{tikz}

% ======================================
% ============= Front page =============
% ======================================
\newcommand{\setlogo}{LogoWordsBottom.png}
\newcommand{\logo}[1]{\renewcommand{\setlogo}{LogoWordsBottom.png}}

\newcommand{\setuniversity}{St. Francis Xavier University}
\newcommand{\university}[1]{\renewcommand{\setuniversity}{#1}}

\newcommand{\setdepartment}{Department of Computer Science}
\newcommand{\department}[1]{\renewcommand{\setdepartment}{#1}}

\newcommand{\setcourse}{}
\newcommand{\course}[1]{\renewcommand{\setcourse}{#1}}

\newcommand{\setcourseshort}{}
\newcommand{\courseshort}[1]{\renewcommand{\setcourseshort}{#1}}

\newcommand{\setexam}{}
\newcommand{\exam}[1]{\renewcommand{\setexam}{#1}}

\newcommand{\StudentID}{\enspace\hrulefill}
\newcommand{\StudentName}{\enspace\hrulefill}

\renewcommand{\maketitle}{
    \begin{center}
    {
        \includegraphics[width=0.7\textwidth]{\setlogo}\\
        \vspace{0.5in}
        \large \textbf{
            \setuniversity \\
            \setdepartment \\
            \vspace{0.1in}
            \setcourse \\
            \setexam \\
            \vspace{0.5in}
        }
    }
    \end{center}

    \makebox[\textwidth][l]{Name: \StudentName}

    \vspace{5mm}

    \makebox[\textwidth][l]{Student ID: \StudentID}
    \vspace{5mm}
}

\newsavebox{\mybox}
\newenvironment{rules}
{
    \begin{center}
    \begin{lrbox}
    {\mybox}\begin{minipage}{\textwidth}
}
{
    \end{minipage}
    \end{lrbox}\fbox{\fbox{\parbox{\textwidth}{\vspace{2mm}\centering\usebox{\mybox}\vspace{2mm}}}}
    \end{center}
}

\newcommand{\showgrade}{
    \begin{center}%
        \gradetable[h][questions]%
    \end{center}%
}

% ======================================
% ========== Header && Footer ==========
% ======================================

\newcommand{\setdate}{}
\newcommand{\examdate}[1]{\renewcommand{\setdate}{#1}}

\runningheader{\setcourseshort} {\setexam} {\setdate}
\runningheadrule
\runningfooter{}{}{Page \thepage\ of \numpages}
\runningfootrule
