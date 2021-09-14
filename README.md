# PhD-Thesis
This repository holds all documents, images, code, and auxillary files that are used in creating ThesisWG270.pdf.
It is standalone, and contains all files required to assemble the thesis `.pdf` file from `ThesisWG270.tex` and `makefile`.

### File Structure
This is seperate from PhD-Work-Git repository to avoid repository bloat; as such, it should only contain the following types of files:

- **`ThesisWG270.tex`**: master thesis file, and target of `latexmk`. This file _is_ the thesis, and imports (precisely, `\inputs`) the other `.tex` files to create the thesis.
- **`baththesis.sty`**: Style file that conforms to the UoB thesis requirements. Manual adjustments to page layout, etc, can be made via editing this file.
- **`makefile`**: Makefile to build the thesis from the command line using `latexmk`. See below for a list of targets.
- `.tex` files that directly contribute content to the thesis. These are stored in the `Chapters` folder. The structure established is to have a sub-folder for each individual chapter of the thesis, containing each _section_ of said chapter in its own `.tex` file. Each folder also contains a `.tex` file which starts the chapter (typically sharing the name of the chapter sub-folder) and contains any text which comes prior to the start of the first section.
- `.tex` files that contain TikZ diagrams. These are stored in the `Diagrams` folder. The diagram `.pdf`s can be produced from the `makefile`, and are placed into the sub-folder `Diagram_PDFs`.
- `.tex` files containing preamble. These are stored in the `Preamble` folder and contain either preamble for thesis prose (including macros defining notation for reoccuring objects) or for TikZ diagrams (which are indirectly used in the `.tex` files for TikZ-produced images). Any established notations or conventions should be placed into the preamble files.
- `.bib` files containing reference information. These are stored in the `BibFiles` folder, as multiple `.bib` files. `ThesisWG270.tex` reads in all of these files to produce one bibliography.

**Still to be included**
- Numerical code and generated diagrams folder and convention

### `makefile` Targets
The `makefile` has the following targets.
- `all`: Produces `ThesisWG270.pdf` from scratch, and cleans up afterwards. Re-renders all supporting TikZ diagrams, but does not re-run numerical code for plots.
- `tikz`: Produces all TikZ diagrams from files stored in `Diagrams` folder, and outputs PDFs to `Diagrams/Diagram_PDFs`.
- `no_tikz`: Produces `ThesisWG270.pdf` without re-rendering any TikZ diagrams.
- `clearfigs`: Removes all `.pdf` files stored in `Diagrams/Diagram_PDFs` (that is, all TikZ diagram outputs).
- `clear`: Soft-cleans the directory. Retains àuxillary files used in the assembly of `ThesisWG270.pdf`. Things like equation numbers, bibtex references, will not be removed which potentially speeds up rendering done through an editor program.
- `clean`: Hard-cleans the directory. Removes all auxillary files created in the previous assembly of `ThesisWG270.pdf`.
