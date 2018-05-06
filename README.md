DFT-FE : Density Functional Theory With Finite-Elements 
=======================================================


About
-----

DFT-FE is a C++ code for material modeling from first principles using Kohn-Sham density functional theory.
It is based on adaptive finite-element based methodologies and handles all-electron and pseudopotential calculations in the 
same framework while accomodating arbitrary boundary conditions. dft-fe code builds on top of the deal.II library for everything 
that has to do with finite elements, geometries, meshes, etc., and, through deal.II on p4est for parallel adaptive mesh handling. 



Installation instructions
-------------------------

The steps to install the necessary dependencies and DFT-FE itself are described
in the Installation instructions section of the DFT-FE manual(compile /doc/manual/manual.tex).



Running DFT-FE
--------------

Instructions on how to run and DFT-FE can also be found in the DFT-FE manual(compile /doc/manual/manual.tex). 



Contributing to DFT-FE
----------------------




More information
----------------

For more information see:

 - The official website at (give link)
 
 - The current manual(compile /doc/manual/manual.tex)

 - For questions about the source code of DFT-FE, portability, installation, etc., use the DFT-FE development mailing list (dft-fe.users@umich.edu).
 
 - DFT-FE is primarily based on the deal.II library. If you have particular questions about deal.II, contact the [deal.II mailist lists](https://www.dealii.org/mail.html).
 
 - If you have specific questions about DFT-FE that are not suitable for the public and archived mailing lists, you can contact the principal developers and mentors:

    - Phani Motamarri: phanim@umich.edu
    - Sambit Das: dsambit@umich.edu
    - Vikram Gavini: vikramg@umich.edu (Mentor)



License
-------

DFT-FE is published under [LGPL v2.1 or newer](LICENSE).