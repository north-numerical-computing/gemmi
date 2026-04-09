addpath(pwd, '-begin');
results = runtests('test_gemmi.m');
disp(table(results));
assert(all([results.Passed]));
