Annotations of semantic conflict in the movie "Forrest Gump"
============================================================

A detailed description of this dataset can be found in the open-access
article:

  Hanke M and Ibe P. (2016) Lies, irony, and contradiction — an annotation of semantic conflict
  in the movie "Forrest Gump". *F1000Research*, 5:237.
  http://f1000research.com/articles/5-2375

All data is released under the terms of the CC0 license.


Content
-------

This repository contains the raw data, the manuscript sources, and
the code to generate summary statistics, and raw figures for the
manuscript, as well as all derived data contained in the data release.

Makefiles are include to execute the associated code. If you are impatient,
just run::

  make

Otherwise read on...

Manuscript
----------

Run ``make paper.pdf`` to compile a PDF of the manuscript (need Latex).
This uses pre-computed summary statistics and hand-optimized figures
that correspond to the published version of the manuscript. See
the next section on how to generate these components.

Derived data and statistics
---------------------------

Run ``make data`` to recompute summary statistics for inclusion in the
manuscript. This command will regenerate the file ``paper/results_def.tex``,
and a subsequent ``make paper.pdf`` will generate an updated manuscript
that reflects any potential changes.

``make data`` will also generate raw SVG files for all figures in the
manuscript under ``paper/figures/generated``. This SVG only differ
from the final figures by layout optimizations and minor improvements
of the visual appearance (done in Inkscape).


