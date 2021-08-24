function x = fistaOverlap(A, b, lambda, groupIndicator, x0)
%FISTAOVERLAP algorithm for the overlap group norm.
%   x = FISTAOVERLAP(A, b, lambda, groupIndicator, x0) finds the minimum point of
%       0.5*||Ax-b||_2^2 + lambda * sum(||x_g||_2)
%   by means of the FISTA algorithm. Here, ||x_g||_2 is the l2 norm of a
%   subvector of x with indices in g. Subvectors may overlap.
%   Input groupIndicator is a logical matrix of size nGroups X length(x): element (i,j)
%   is true iff x(j) belongs to group i. Input x0 (optional) is the starting
%   point of the iteration sequence.

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
    x_new = lambda / L * proximalOverlap(tmp * L / lambda, groupIndicator);
    delta = norm(x_old-x_new);
    %fprintf('%f\n',delta);
    t_old = t_new;
    t_new = (1 + sqrt(1 + 4*t_old^2))/2;
    y = x_new + (t_old - 1)/t_new * (x_new - x_old);
end
x = x_new;
