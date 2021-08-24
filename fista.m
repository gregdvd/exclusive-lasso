function x = fista(A, b, lambda, groups, x0)
%FISTA algorithm for the group-based l2(l1) norm.
%   x = FISTA(A, b, lambda, groups, x0) finds the minimum point of
%       0.5*||Ax-b||_2^2 + lambda * ||x||_l2(l1)
%   by means of the FISTA algorithm. Here, ||x||_l2(l1) is the l2 norm of a
%   vector of positive reals obtained by computing the l1 norm of all groups.
%   Input groups is a cell of arrays, each array containing the indices of a
%   different group. Input x0 (optional) is the starting point of the iteration
%   sequence.

MAX_ITER = 500;
THRESHOLD = 1e-4;

nFeatures = length(A(1,:));
if nargin == 4
    x0 = zeros(nFeatures, 1);
end
e = eig(A'*A);
L = max(e); % smallest Lipschitz constant
y = x0;
x_new = zeros(nFeatures, 1);
x_old = x0;
t_new = 1;
delta = 1;
thisIter = 0;
while delta > THRESHOLD * norm(x_old) && thisIter < MAX_ITER
    thisIter = thisIter + 1;
    x_old = x_new;
    tmp = y - A'*(A*y - b)/L;
    x_new = lambda / L * proximal(tmp * L / lambda, groups);
    delta = norm(x_old-x_new);
    %fprintf('%f\n',delta);
    t_old = t_new;
    t_new = (1 + sqrt(1 + 4*t_old^2))/2;
    y = x_new + (t_old - 1)/t_new * (x_new - x_old);
end
x = x_new;
