%  GEMMI    Compute matrix product using integer Ozaki scheme.
%   [C, ALGOUT] = GEMMI(A,B,ASPLITS,BSPLITS,ALGIN) computes the matrix
%   C = A * B using the Ozaki scheme with ASPLITS and BSPLITS slices
%   for the matrices A and B, respectively. The ALGIN parameter
%   must be a struct, with the following fields currently supported.
%   'split' - selects the stragegy to be used to split A and B into
%             slices. Possible values are 'b' for bitmasking and 'n'
%             for round-to-nearest (default).
%   'mult'  - selects how many integer multiplications the algorithm
%             will perform in order to compute the result. Possible
%             values are 'a' for all ASPLIT * BSPLIT products and 'r'
%             for a reduced number (default).
%   'acc'   - selects how the exact integer matrix products are
%             accumulated. Possible values are 'f' for floating-point
%             arithmetic and 'i' for integer accumulation (default).
%   The output paramater ALGOUT is a struct with the same fields as
%   ALGIN, which contains the values used in the computation.
%
%   [...] = GEMMI(A,B,ASPLITS,BSPLITS) uses the ALGIN parameter passed
%   the most recent call to GEMMI, or the default values if no previous
%   call was made.
%
%   [...] = GEMMI(A,B,SPLITS) uses SPLITS slices for both A and B.
%
%   The splits are stored as 8-bit signed integer, the dot products are
%   performed using 32-bit signed arithmetic, and the final accumulation
%   uses either the same format as the matrices A and B (if 'acc' is 'f')
%   or 32-bit arithmetic (if 'acc' is 'i').
%
%   The matrices A and B must be conformable, and multiplication by a
%   scalar is not supported.
