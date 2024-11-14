%  GEMMI    Compute matrix product using integer Ozaki scheme.
%   C = GEMMI(A,B,ASPLITS,BSPLITS) computes the product A*B using
%   the Ozaki scheme with ASPLITS slices for the matrix A and BSPLITS
%   slices for the matrix B.
%
%   C = GEMMI(A,B,SPLITS) uses SPLITS slices for both A and B.
% 
%   The splits are stored as 8-bit signed integer, the dot products are
%   performed using 32-bit signed arithmetic, and the final accumulation
%   uses either binary32 or binary64 arithmetic, depending on the type
%   of A and B.
%
%   The matrices A and B must be conformable, and multiplication by a
%   scalar is not supported.