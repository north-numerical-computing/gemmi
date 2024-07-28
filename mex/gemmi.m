%GEMMI    Compute matrix product using integer Ozaki scheme.
%   C = GEMMI(A,B,S) computes the product A*B using the Ozaki scheme
%   with S splits. The splits are stored as 8-bit signed integer, the
%   dot products are performed using 32-bit signed arithmetic, and the
%   final accumulation uses either binary32 or binary64 arithmetic,
%   depending on the type of A and B.

%   The matrices A and B must be conformable, and multiplication by a
%   scalar is not supported.
