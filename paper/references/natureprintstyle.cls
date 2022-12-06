%% Class natureprintstyle
%% Written by Brendon Higgins, blhiggins@gmail.com
%% v1.0 - 19 April 2008
%% Based on class nature, written by Peter Czoschke, czoschke@mrl.uiuc.edu
%% V1.1 - 13 February 2012
%% Various minor changes by Michele Cappellari (Oxford) to agree more 
%% closely with the style of published Nature articles.
%%
%% A drop-in replacement for the nature.cls class, used for Nature letter and
%% article preprints, that approximates Nature's style in print.
%%
%% Note that \includegraphics commands are not ignored, as they are in
%% nature.cls.
%%
%% This class may be particularly handy if an author finds it annoying to
%% read the single column format of the Nature preprint style.
%%
%% I created this class for personal purposes and without any connection
%% to the Nature Publishing Group.  I in no way claim that documents generated
%% with this file fully comply with their current style requirements.
%% I disclaim any responsibility for the use of this file heretofore.
%%
%% ------------------------------------
%%
%% See nature.cls and its documentation for notes on usage.
%%
%% The file nature-template.tex, an example for the nature.cls class, should
%% also work with this class. Simply replace \documentclass{nature} with
%% \documentclass{natureprintstyle}.

\ProvidesClass{natureprintstyle}[13/2/2012 v1.1]
% \typeout{A class for emulating the style of Nature in print when writing preprints for the journal Nature}
% \typeout{Written by Brendon Higgins}
% \typeout{Based on nature.cls by Peter Czoschke}
% \typeout{ }

\LoadClass[10pt,twocolumn]{extarticle}
\RequirePackage{times}
\RequirePackage{cite}
\RequirePackage{ifthen}
\RequirePackage[total={18.2cm,24.4cm},centering]{geometry}
\RequirePackage{scalefnt}
\RequirePackage{type1cm}
\RequirePackage{color}
\definecolor{nouvbl}{rgb}{0.,0.,0.55}

%% make labels in bibliobraphy be #.
\renewcommand\@biblabel[1]{#1.}

\def\bibcommenthead{}%

%% make citations be superscripts, taken from citesupernumber.sty
\def\@cite#1#2{$^{\mbox{\scriptsize #1\if@tempswa , #2\fi}}$}

%% Some style parameters
\setlength{\parindent}{0.20in}
\newcommand{\spacing}[1]{\renewcommand{\baselinestretch}{#1}\large\normalsize}

%% Redefine \maketitle for Nature style
\def\@maketitle{%
  \newpage\spacing{1}\setlength{\parskip}{3pt}%
    {\color{nouvbl} \fontsize{40}{10}\selectfont LETTER\par \rule{\textwidth}{3pt} \par}%
    {\scalefont{2.7}\noindent\sloppy%
        \begin{flushleft}\bfseries\@title\end{flushleft} \par}%
    {\scalefont{1.1}\noindent\sloppy \@author \par \vspace{0.5cm}}%
}

%% Define the affiliations environment, list each institution as an \item
%% Put as footnote of first paragraph
\newenvironment{affiliations}{%
    \let\olditem=\item
    \renewcommand\item[1][]{$^{\arabic{enumi}}$\stepcounter{enumi}}
    \setcounter{enumi}{1}%
    \setlength{\parindent}{0in}%
    \sffamily\sloppy%
    \scalefont{0.83}
    }{\let\item=\olditem}

%% Redefine the abstract environment to be the first bold paragraph
\renewenvironment{abstract}{%
    \setlength{\parindent}{0in}%
    \setlength{\parskip}{0in}%
    \bfseries%
    }{\par}

%% Redefine the \section command.
\renewcommand{\section}{\@startsection {section}{1}{0pt}%
    {12pt}{1pt}%
    {\sffamily\bfseries\scalefont{1.1}}%
    }
\renewcommand{\subsection}{\@startsection {subsection}{2}{0pt}%
    {0pt}{-0.5em}%
    {\bfseries}%
    }

%% Define the methodssummary environment.  Use \subsection to separate. These come before methods.
\newenvironment{methodssummary}{%
    \section*{METHODS SUMMARY}%
    \setlength{\parskip}{0pt}%
    \scalefont{0.93}
    }{}

%% Define the methods environment.  Use \subsection to separate.
\newenvironment{methods}{%
    \section*{METHODS}%
    \setlength{\parskip}{0pt}%
    \scalefont{0.93}
    }{}

%% No heading for References section, but eat up the extra space from \section command
\renewcommand\refname{}

\let\oldthebibliography=\thebibliography
  \let\endoldthebibliography=\endthebibliography
  \renewenvironment{thebibliography}[1]{%
    \begin{oldthebibliography}{#1}%
      \sffamily%
      \scalefont{0.83}%
      \setlength{\parskip}{-4pt}%
  }%
  {%
    \end{oldthebibliography}%
  }

\let\oldbibitem=\bibitem
\renewcommand{\bibitem}[1]{\vspace{-0.15ex}\oldbibitem{#1}}
%% bibitem takes an optional parameter, so this might be broken.

\def\newblock{\hskip .11em plus .33em minus -.07em}
% \sfcode`\.=1000\relax


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %citations - note the nature.bst bibliograpy style must be used.
% % the #1 is the name
% % the #2 is the reference number
% \def\bibitem[#1<#2>]#3{\item[#2]
%     \if@filesw
% 	{\def\protect##1{\string ##1\space}
% 	\immediate\write\@auxout{\string\lbibcite{#3}{#1}{#2}}}
%     \fi}
% %note that the lbibcite defines two types of citation
% % b@ is the name
% % by@ is the number
% \def\lbibcite#1#2#3{\global\@namedef{b@#1}{#2}
%                     \global\@namedef{by@#1}{#3}}
% 
% % redefine the citation to make a superscript
% 
% %\def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
% %  \def\@citea{}\@cite{\@for\@citeb:=#2\do
% %    {\@citea\def\@citea{$^,$}\@ifundefined
% %       {b@\@citeb}{{\bf$^?$}\@warning
% %       {Citation `\@citeb' on page \thepage \space undefined}}
% %{$^{\csname by@\@citeb\endcsname}$}}}
% %{#1}}
% 
% % Collapse citation numbers to ranges.  Non-numeric and undefined labels
% % are handled.  No sorting is done.  E.g., 1,3,2,3,4,5,foo,1,2,3,?,4,5
% % gives 1,3,2-5,foo,1-3,?,4,5
% % stolen from aip.sty and adapted pah - 7/08/92
% \newcount\@tempcntc
% \def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
%   \@tempcnta\z@\@tempcntb\m@ne\def\@citea{}\@cite{\@for\@citeb:=#2\do
%     {\@ifundefined
%        {by@\@citeb}{\@citeo\@tempcntb\m@ne\@citea\def\@citea{,}{\bf ?}\@warning
%        {Citation `\@citeb' on page \thepage \space undefined}}%
%     {\setbox\z@\hbox{\global\@tempcntc0\csname by@\@citeb\endcsname\relax}%
%      \ifnum\@tempcntc=\z@ \@citeo\@tempcntb\m@ne
%        \@citea\def\@citea{,}\hbox{\csname by@\@citeb\endcsname}%
%      \else
%       \advance\@tempcntb\@ne
%       \ifnum\@tempcntb=\@tempcntc
%       \else\advance\@tempcntb\m@ne\@citeo
%       \@tempcnta\@tempcntc\@tempcntb\@tempcntc\fi\fi}}\@citeo}{#1}}
% \def\@citeo{\ifnum\@tempcnta>\@tempcntb\else\@citea\def\@citea{,}%
%   \ifnum\@tempcnta=\@tempcntb\the\@tempcnta\else
%    {\advance\@tempcnta\@ne\ifnum\@tempcnta=\@tempcntb \else \def\@citea{--}\fi
%     \advance\@tempcnta\m@ne\the\@tempcnta\@citea\the\@tempcntb}\fi\fi}
% 
% 
% 
% %\def\@cite#1#2{#1\if@tempswa
% %	\@warning{notes not allowed in citations for NATURE}\fi}
% 
% % Superscript citations -- skip optional arg to \cite
% % Move citation after period and comma.
% \def\@cite#1#2{\unskip\nobreak\relax
%     \def\@tempa{$\m@th^{\hbox{\the\scriptfont0 #1}}$}%
%     \futurelet\@tempc\@citexx}
% \def\@citexx{\ifx.\@tempc\let\@tempd=\@citepunct\else
%     \ifx,\@tempc\let\@tempd=\@citepunct\else
%     \let\@tempd=\@tempa\fi\fi\@tempd}
% \def\@citepunct{\@tempc\edef\@sf{\spacefactor=\the\spacefactor\relax}\@tempa
%     \@sf\@gobble}
% 
% % other citations
% \def\scite#1{\if@filesw\immediate\write\@auxout{\string\citation{#1}}\fi
%        \expandafter\ifx\csname b@#1\endcsname\relax
%        {{\bf$^?$}\@warning
%        {Citation `#1' on page \thepage \space undefined}}\else
% {\csname b@#1\endcsname $^{\csname by@#1\endcsname}$}\fi
% }
% 
% \def\pcite#1{\if@filesw\immediate\write\@auxout{\string\citation{#1}}\fi
%        \expandafter\ifx\csname b@#1\endcsname\relax
%        {{\bf$^?$}\@warning
%        {Citation `#1' on page \thepage \space undefined}}\else
% {\csname by@#1\endcsname}\fi
% }
% 
% \def\thebibliography#1
% {\vskip \baselineskip
% \hrule
% \small
% \vskip 3pt
% \@dates
% \list{}{
% \settowidth\labelwidth{#1.}
% \itemindent 0pt\labelsep 5pt
% \leftmargin \labelwidth
% \addtolength{\leftmargin}{5pt}
% \renewcommand{\makelabel}[1]{##1.}
%   }
% \def\newblock{\hskip .11em plus .33em minus -.07em}
% \sfcode`\.=1000\relax
% }
% \def\endthebibliography{\endlist\hrule}
% % end of Citations 



%% Define the addendum environment for Supplementary Info, Acknowledgements, etc.
\newenvironment{addendum}{%
    \setlength{\parindent}{0in}%
    \sffamily%
    \scalefont{0.83}%
    \begin{list}{Acknowledgements}{%
        \setlength{\leftmargin}{0in}%
        \setlength{\listparindent}{0in}%
        \setlength{\labelsep}{0em}%
        \setlength{\labelwidth}{0in}%
        \setlength{\itemsep}{5pt}%
        \let\makelabel\addendumlabel}
    }
    {\end{list}\par\vfill}

\newcommand*{\addendumlabel}[1]{\textbf{#1}\hspace{1em}}

%% Figures and tables:
%% The \includegraphics command is respected.
%% Tables should all be moved to the end of the document
%% manually.

\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}

%% Change the way captions are formatted. Based on a tip by Jakob Schiotz.
%% http://dcwww.camd.dtu.dk/~schiotz/comp/LatexTips/LatexTips.html
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\textbf{\scalefont{0.93}#1 $\vert$}} \scalefont{0.93}#2}%
  \ifdim \wd\@tempboxa >\hsize
    {\scalefont{0.93}{\textbf{\scalefont{0.93}#1 $\vert$}} \scalefont{0.93}#2\par}
  \else
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%% ADS journal abbreviations
\let\jnl@style=\sffamily 
\def\ref@jnl#1{{\jnl@style#1}}% 
\newcommand\aj{\ref@jnl{AJ}}%        % Astronomical Journal 
\newcommand\araa{\ref@jnl{ARA\&A}}%  % Annual Review of Astron and Astrophys 
\newcommand\apj{\ref@jnl{ApJ}}%    % Astrophysical Journal 
\newcommand\apjl{\ref@jnl{ApJL}}     % Astrophysical Journal, Letters 
\newcommand\apjs{\ref@jnl{ApJS}}%    % Astrophysical Journal, Supplement 
\newcommand\ao{\ref@jnl{ApOpt}}%   % Applied Optics 
\newcommand\apss{\ref@jnl{Ap\&SS}}%  % Astrophysics and Space Science 
\newcommand\aap{\ref@jnl{A\&A}}%     % Astronomy and Astrophysics 
\newcommand\aapr{\ref@jnl{A\&A~Rv}}%  % Astronomy and Astrophysics Reviews 
\newcommand\aaps{\ref@jnl{A\&AS}}%    % Astronomy and Astrophysics, Supplement 
\newcommand\azh{\ref@jnl{AZh}}%       % Astronomicheskii Zhurnal 
\newcommand\baas{\ref@jnl{BAAS}}%     % Bulletin of the AAS 
\newcommand\icarus{\ref@jnl{Icarus}}% % Icarus
\newcommand\jaavso{\ref@jnl{JAAVSO}}  % The Journal of the American Association of Variable Star Observers
\newcommand\jrasc{\ref@jnl{JRASC}}%   % Journal of the RAS of Canada 
\newcommand\memras{\ref@jnl{MmRAS}}%  % Memoirs of the RAS 
\newcommand\mnras{\ref@jnl{MNRAS}}%   % Monthly Notices of the RAS 
\newcommand\pra{\ref@jnl{PhRvA}}% % Physical Review A: General Physics 
\newcommand\prb{\ref@jnl{PhRvB}}% % Physical Review B: Solid State 
\newcommand\prc{\ref@jnl{PhRvC}}% % Physical Review C 
\newcommand\prd{\ref@jnl{PhRvD}}% % Physical Review D 
\newcommand\pre{\ref@jnl{PhRvE}}% % Physical Review E 
\newcommand\prl{\ref@jnl{PhRvL}}% % Physical Review Letters 
\newcommand\pasp{\ref@jnl{PASP}}%     % Publications of the ASP 
\newcommand\pasj{\ref@jnl{PASJ}}%     % Publications of the ASJ 
\newcommand\qjras{\ref@jnl{QJRAS}}%   % Quarterly Journal of the RAS 
\newcommand\skytel{\ref@jnl{S\&T}}%   % Sky and Telescope 
\newcommand\solphys{\ref@jnl{SoPh}}% % Solar Physics 
\newcommand\sovast{\ref@jnl{Soviet~Ast.}}% % Soviet Astronomy 
\newcommand\ssr{\ref@jnl{SSRv}}% % Space Science Reviews 
\newcommand\zap{\ref@jnl{ZA}}%       % Zeitschrift fuer Astrophysik 
\newcommand\nat{\ref@jnl{Nature}}%  % Nature 
\newcommand\iaucirc{\ref@jnl{IAUC}}% % IAU Cirulars 
\newcommand\aplett{\ref@jnl{Astrophys.~Lett.}}%  % Astrophysics Letters 
\newcommand\apspr{\ref@jnl{Astrophys.~Space~Phys.~Res.}}% % Astrophysics Space Physics Research 
\newcommand\bain{\ref@jnl{BAN}}% % Bulletin Astronomical Institute of the Netherlands 
\newcommand\fcp{\ref@jnl{FCPh}}%   % Fundamental Cosmic Physics 
\newcommand\gca{\ref@jnl{GeoCoA}}% % Geochimica Cosmochimica Acta 
\newcommand\grl{\ref@jnl{Geophys.~Res.~Lett.}}%  % Geophysics Research Letters 
\newcommand\jcp{\ref@jnl{JChPh}}%     % Journal of Chemical Physics 
\newcommand\jgr{\ref@jnl{J.~Geophys.~Res.}}%     % Journal of Geophysics Research 
\newcommand\jqsrt{\ref@jnl{JQSRT}}%   % Journal of Quantitiative Spectroscopy and Radiative Trasfer 
\newcommand\memsai{\ref@jnl{MmSAI}}% % Mem. Societa Astronomica Italiana 
\newcommand\nphysa{\ref@jnl{NuPhA}}%     % Nuclear Physics A 
\newcommand\physrep{\ref@jnl{PhR}}%       % Physics Reports 
\newcommand\physscr{\ref@jnl{PhyS}}%        % Physica Scripta 
\newcommand\planss{\ref@jnl{Planet.~Space~Sci.}}%  % Planetary Space Science 
\newcommand\procspie{\ref@jnl{Proc.~SPIE}}%      % Proceedings of the SPIE 

\newcommand\actaa{\ref@jnl{AcA}}%  % Acta Astronomica
\newcommand\caa{\ref@jnl{ChA\&A}}%  % Chinese Astronomy and Astrophysics
\newcommand\cjaa{\ref@jnl{ChJA\&A}}%  % Chinese Journal of Astronomy and Astrophysics
\newcommand\jcap{\ref@jnl{JCAP}}%  % Journal of Cosmology and Astroparticle Physics
\newcommand\na{\ref@jnl{NewA}}%  % New Astronomy
\newcommand\nar{\ref@jnl{NewAR}}%  % New Astronomy Review
\newcommand\pasa{\ref@jnl{PASA}}%  % Publications of the Astron. Soc. of Australia
\newcommand\rmxaa{\ref@jnl{RMxAA}}%  % Revista Mexicana de Astronomia y Astrofisica

%% added feb 9, 2016
\newcommand\maps{\ref@jnl{M\&PS}}% Meteoritics and Planetary Science
\newcommand\aas{\ref@jnl{AAS Meeting Abstracts}}% American Astronomical Society Meeting Abstracts
\newcommand\dps{\ref@jnl{AAS/DPS Meeting Abstracts}}% American Astronomical Society/Division for Planetary Sciences Meeting Abstracts
