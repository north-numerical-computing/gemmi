function retval = compile_mex(varargin)
%OMPILE_MEX    Compile MEX interface to GEMMI.

  include_dir = '';
  coptions = '-std=c++20 -O3 -march=native';
  clibs = '-lm';

  usingoctave = exist('OCTAVE_VERSION', 'builtin');
  if usingoctave
    setenv("CFLAGS", sprintf("%s", coptions));
    libpath = deblank(evalc('mkoctfile --print OCTLIBDIR'));
    setenv("LDFLAGS", sprintf("-fopenmp %s -L%s", clibs, libpath));
    [~, status] = mkoctfile('gemmi.cpp', '--mex', '--verbose');
  else
      mex('gemmi.cpp', '-silent',...
          include_dir,...
          [sprintf('CFLAGS=$CFLAGS %s  ', coptions)],...
          [sprintf('LDFLAGS=$LDFLAGS %s ', clibs)]);
  end
end
