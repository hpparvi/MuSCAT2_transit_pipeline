Pre-analysis steps
------------------

0. Create a directory where you will store all the MuSCAT2 analyses. This isn't strictly necessary, but makes life
   easier in the future. we'll call this directory `M2ROOT` from now on, but the name doesn't matter.

1. Execute `m2init` in the root directory to create an analysis directory `<target_name>` with the
   default directory structure

   .. code-block:: bash

        > cd M2ROOT
        > m2init <target_name>

2. Copy the photometry from each night to `<target_name>/photometry/<yymmdd>` subdirectories.
3. Move into the analysis directory and execute `m2nbtemplate` to create a template notebook

   .. code-block:: bash

        > cd <target_name>
        > m2nbtemplate <target_name> <yymmdd>

4. Open the template notebook in Jupyter. It's a good practice to run a Jupyter notebook server from the
   `M2ROOT` so that you don't need to start a new server for each analyses. So, if you don't have a server
   running, do

   .. code-block:: bash

        > cd M2ROOT
        > jupyter notebook

   but if you already have a server running, use it.