function p = proximalOverlap(x, groupIndicator)
%PROXIMALOVERLAP proximal operator for the overlap group lasso.
%   p = PROXIMALOVERLAP(x, groupIndicator) computes the proximal point p of vector
%   x according to the overlap group lasso norm. x is a vector of reals.
%   groupIndicator is a logical matrix of size (nGroups x length(x)): element (i,j)
%   is true iff x(j) belongs to group i.

TOL = 1e-5;
ArmijoBeta = 0.5;
ArmijoSigma = 1e-4;

nTotalEntries = length(x);
nGroups = size(groupIndicator, 1);
assert(nTotalEntries == size(groupIndicator, 2))

% Groups with Euclidean norm not larger than 1 are not meaningful.
activeGroups = false(nGroups, 1);
for iGroup = 1:nGroups
    activeGroups(iGroup) = (norm(x(groupIndicator(iGroup, :))) > 1);
end

groupIndicatorPurged = groupIndicator(activeGroups, :);
nGroupsPurged = size(groupIndicatorPurged, 1);

% The proximal point is given by x - PI(x), with PI(x) the projection of x to
% the unitary ball of the dual norm. Thus, we start by computing PI(x). This
% can be done by solving the dual problem by Bertsekas' projected Newton method.
isSolved = false;
iIter = 0;
lambdas = zeros(nGroupsPurged, 1); % the unknown dual variable
allIndices = 1:nTotalEntries;

gradf = computeGradient;

while ~isSolved
    iIter = iIter + 1;
    currentf = computeObjective(lambdas);

    thisTol = min(TOL, norm(lambdas - max(lambdas - gradf, 0)));
    activeCnstrs = ( (lambdas >= 0) & (lambdas <= thisTol) & (gradf > 0) );
    inactiveCnstrs = ~activeCnstrs;
    hessian = nan(nGroupsPurged);

    for iGroup = 1:nGroupsPurged
        for jGroup = 1:iGroup
            if iGroup == jGroup || inactiveCnstrs(iGroup) && inactiveCnstrs(jGroup)
                intersection = groupIndicatorPurged(iGroup, :) & groupIndicatorPurged(jGroup, :);
                interIndices = allIndices(intersection);
                tmp = 0;
                for thisIndex = interIndices
                    tmp = tmp + 2 * x(thisIndex)^2 / (1 + sum(lambdas(groupIndicatorPurged(:, thisIndex))))^3;
                end
                hessian(iGroup, jGroup) = tmp;
                hessian(jGroup, iGroup) = tmp;
            else
                hessian(iGroup, jGroup) = 0;
                hessian(jGroup, iGroup) = 0;
            end
        end
    end

    hessInvGrad = (thisTol * eye(nGroupsPurged) + hessian) \ gradf;
    armijoExp = 1;
    lambdasTmp = lambdas - ArmijoBeta * hessInvGrad; % Newton step
    lambdasTmp = lambdasTmp .* (lambdasTmp > 0); % projection onto the nonnegative orthant

    % Use an Armijo-like rule to select a proper step-size.
    while ~isArmijoOK(armijoExp, lambdasTmp)
        armijoExp = armijoExp + 1;
        lambdasTmp = lambdas - ArmijoBeta^armijoExp * hessInvGrad;
        lambdasTmp = lambdasTmp .* (lambdasTmp > 0);
    end

    lambdas = lambdasTmp;

    gradf = computeGradient;
    exitConditions = false(nGroupsPurged, 1);
    % Check whether the first-order necessary conditions are met for lambdas to
    % be a constrained local minimum point.
    exitConditions(lambdas == 0) = ( gradf(lambdas == 0) > 0 );
    exitConditions(lambdas > 0) = ( abs(gradf(lambdas > 0)) <= TOL );
    isSolved = all(exitConditions);

end

% Return the proximal point.
p = x - computeProjection;


    function f = computeObjective(lambdas_)
    % Compute the value of the objective function.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    f = sum(lambdas_);
    for iEntry_ = 1:nTotalEntries
        f = f + x(iEntry_)^2 / (1 + sum(lambdas_(groupIndicatorPurged(:, iEntry_))));
    end
    end


    function df = computeGradient
    % Compute the gradient of the objective function at the current point.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    df = ones(nGroupsPurged, 1);
    for iGroup_ = 1:nGroupsPurged
        groupIndices_ = allIndices(groupIndicatorPurged(iGroup_, :));
        for thisIndex_ = groupIndices_
            df(iGroup_) = df(iGroup_) - x(thisIndex_)^2 / (1 + sum(lambdas(groupIndicatorPurged(:, thisIndex_))))^2;
        end
    end
    end


    function check = isArmijoOK(armijoExp_, lambdasTmp_)
    % Check whether the step size was small enough according to the Armijo-like rule.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    tmp_ = gradf .* hessInvGrad;
    rhs = ArmijoBeta^armijoExp_ * sum(tmp_(inactiveCnstrs));
    term2 = gradf .* (lambdas - lambdasTmp_);
    rhs = rhs + sum(term2(activeCnstrs));
    fTmp = computeObjective(lambdasTmp_);
    check = ( currentf - fTmp >= ArmijoSigma * rhs );
    end


    function proj = computeProjection
    % Compute the projection of x onto the unitary ball of the dual norm by
    % primal-dual feasibility after the dual optimization problem is solved.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    proj = x;
    for ii = 1:nTotalEntries
        proj(ii) = proj(ii) / (1 + sum(lambdas(groupIndicatorPurged(:, ii))));
    end
    end


end
