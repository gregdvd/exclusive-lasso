function x = activestrings(A, b, mu, groups, stringMinLength)
%ACTIVESTRINGS active set algorithm that looks for long strings.
%   x = ACTIVESTRINGS(A, b, mu, groups, s) finds the minimum point of
%   by means of the active-set algorithm. The solution is constrained to consist
%   of strings of at least s active parameters. Here, ||x||_l2(l1) is the l2 norm of a
%   vector of positive reals obtained by computing the l1 norm of all groups.
%   Input groups is a cell of arrays, each array containing the indices of a
%   different group.

DUALITYGAP = 0.01; % max duality gap allowed
THRESHOLD = 1e-5; % minimum allowable value of active entries
MULTIPLESTRINGS = true; % the active set can consists in more than one string
solveRestricted = @solveRestrictedVar;
% Use suffix IP for interior-point, Var for variational formulation.

[nObservations, nFeatures] = size(A);
x0 = zeros(nFeatures, 1);
nGroups = length(groups);
assert(nGroups == stringMinLength, ['Only the case where the minimum string '...
    'length is equal to the number of groups is considered.'])
usedEntries = false(nFeatures, 1);
featureGroup = nan(nFeatures, 1);
tmpString.first = [];
tmpString.last = [];
stringList(nFeatures/stringMinLength) = tmpString;
nActiveStrings = 0;

counts = zeros(nFeatures, 1);
partition(nGroups) = Subset;

for iGroup = 1:nGroups
    featureGroup(groups{iGroup}) = iGroup;
    thisGroup = groups{iGroup};
    counts(thisGroup) = counts(thisGroup) + 1;
    partition(iGroup) = Subset(x0, thisGroup);
    ixActive = thisGroup(abs(x0(thisGroup)) > 0);
    usedEntries(ixActive) = true;
    partition(iGroup).activateIndices(ixActive);
end
assert(all(counts == 1), 'groups should describe a partition of the indices of x');

currentGradient = A' * (A * x0 - b);
[passed, nextActive] = checknecessary(true);
x = x0;
while ~passed && sum(usedEntries) < nObservations/2
    if nextActive.stringIndex <= nActiveStrings
        newIndices = nextActive.index;
        stringIndex = nextActive.stringIndex;
        stringList(stringIndex).first = min(stringList(stringIndex).first, newIndices);
        stringList(stringIndex).last = max(stringList(stringIndex).last, newIndices);
    else
        newIndices = nextActive.index + (0:stringMinLength-1);
        nActiveStrings = nActiveStrings + 1;
        stringList(nActiveStrings).first = newIndices(1);
        stringList(nActiveStrings).last = newIndices(end);
    end

    usedEntries(newIndices) = true;
    for thisIndex = newIndices
        ixGroup = featureGroup(thisIndex);
        ixtmp = partition(ixGroup).getindices('active');
        partition(ixGroup).activateIndices([ixtmp; thisIndex]);
    end
    [stringList, nActiveStrings] = compactStrings(stringList, nActiveStrings);

    [x, partition] = solveRestricted(A, b, mu, partition, x);
    currentGradient = A' * (A * x - b);
    [passed, nextActive] = checknecessary(MULTIPLESTRINGS);
end

[passed, nextActive] = checksufficient(false);
while ~passed && sum(usedEntries) < nObservations/2
    if nextActive.stringIndex <= nActiveStrings
        newIndices = nextActive.index;
        stringIndex = nextActive.stringIndex;
        stringList(stringIndex).first = min(stringList(stringIndex).first, newIndices);
        stringList(stringIndex).last = max(stringList(stringIndex).last, newIndices);
    else
        newIndices = nextActive.index + (0:stringMinLength-1);
        nActiveStrings = nActiveStrings + 1;
        stringList(nActiveStrings).first = newIndices(1);
        stringList(nActiveStrings).last = newIndices(end);
    end

    usedEntries(newIndices) = true;
    for thisIndex = newIndices
        ixGroup = featureGroup(thisIndex);
        ixtmp = partition(ixGroup).getindices('active');
        partition(ixGroup).activateIndices([ixtmp; thisIndex]);
    end
    [stringList, nActiveStrings] = compactStrings(stringList, nActiveStrings);

    [x, partition] = solveRestricted(A, b, mu, partition, x);
    currentGradient = A' * (A * x - b);
    [passed, nextActive] = checksufficient(false);
end
x = cleanfeatures(x);


% nested functions

    function featOut = cleanfeatures(featIn)
    % Since the algorithm is suboptimal, we need to remove possible isolated active
    % entries outside the main strings.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    featOut = featIn .* (featIn >= THRESHOLD);

    ixList = 1:nFeatures;
    ixUsed = ixList(featOut ~= 0);

    for ii = ixUsed
        ixCheck = max(1, ii-stringMinLength+1):min(nFeatures-stringMinLength+1, ii);
        neighbourhoods = zeros(stringMinLength, 1);

        for jj=1:length(ixCheck)
            neighbourhoods(jj) = sum(featOut(ixCheck(jj)+(0:stringMinLength-1)) ~= 0);
        end
        if all(neighbourhoods < stringMinLength)
            featOut(ii) = 0;
        end
    end
    end

    function [passed, nextActive] = checknecessary(considerMultipleStrings)
    % Check whether the necessary condition for the solution is met and suggest the
    % next entry to activate.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    passed = true;
    maxMargin = -inf;
    nextActive = [];

    for iString = 1:nActiveStrings
        thisString = stringList(iString);
        [tmpMax, tmpIx] = expandStringN(thisString.first, thisString.last);
        passed = passed && (tmpMax <= 0);
        if tmpMax > maxMargin
            maxMargin = tmpMax;
            nextActive.index = tmpIx;
            nextActive.stringIndex = iString;
        end
    end

    if considerMultipleStrings
        for stringStart = 1:nFeatures-stringMinLength+1
            if all(usedEntries(stringStart:stringStart+stringMinLength-1) == false)
                stringMargin = 0;
                for jj = stringStart:stringStart+stringMinLength-1
                    jjGroup = featureGroup(jj);
                    thisPartition = partition(jjGroup);
                    inactiveGradient = currentGradient(jj);
                    localNormOne = thisPartition.getnormone;

                    if localNormOne > 0
                        jjMargin = abs(inactiveGradient) / localNormOne - mu;
                    else
                        jjMargin = abs(inactiveGradient);
                    end
                    stringMargin = stringMargin + jjMargin;
                    passed = passed && (jjMargin <= 0);
                end

                stringMargin = stringMargin / stringMinLength;
                if stringMargin > maxMargin
                    maxMargin = stringMargin;
                    nextActive.index = stringStart;
                    nextActive.stringIndex = nActiveStrings+1;
                end
            end
        end
    end
    end


    function [tmpMax, tmpIx] = expandStringN(first, last)
    % Check towards which end the sequence should expand (for necessary condition)

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    tmpMax = -inf;
    tmpIx = nan;
    extIndices = [first-1, last+1];
    extIndices = extIndices((extIndices >= 1) & (extIndices <= nFeatures));

    for thisIndex_ = extIndices
        extGroup = featureGroup(thisIndex_);
        extPartition = partition(extGroup);
        inactiveGradient = currentGradient(thisIndex_);
        localNormOne = extPartition.getnormone;

        if localNormOne > 0
            extMargin = abs(inactiveGradient) / localNormOne - mu;
        else
            extMargin = abs(inactiveGradient);
            warning('You shouldn''t be here!')
        end

        if extMargin > tmpMax
            tmpMax = extMargin;
            tmpIx = thisIndex_;
        end
    end
    end


    function [passed, nextActive] = checksufficient(considerMultipleStrings)
    % Check whether the sufficient condition for the solution is met and suggest the
    % next entry to activate.

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    maxMargin = -inf;
    nextActive = [];

    % decide what indices must be activated if the check fails
    for iString = 1:nActiveStrings
        thisString = stringList(iString);
        [tmpMax, tmpIx] = expandStringS(thisString.first, thisString.last);
        for ii=1:length(tmpMax)
            if tmpMax(ii) > maxMargin
                maxMargin = tmpMax(ii);
                nextActive.index = tmpIx(ii);
                nextActive.stringIndex = iString;
            end
        end
    end

    if considerMultipleStrings
        for stringStart = 1:nFeatures-stringMinLength+1
            if all(usedEntries(stringStart:stringStart+stringMinLength-1) == false)
                stringMargin = 0;
                for jj = stringStart:stringStart+stringMinLength-1
                    jjGroup = featureGroup(jj);
                    thisPartition = partition(jjGroup);
                    inactiveGradient = currentGradient(jj);

                    if isempty(thisPartition.getindices('active'))
                        assert(inactiveGradient == 0);
                        continue;
                    end

                    jjMargin = inactiveGradient^2;
                    stringMargin = stringMargin + jjMargin;
                end

                stringMargin = stringMargin / stringMinLength;
                if stringMargin > maxMargin
                    maxMargin = stringMargin;
                    nextActive.index = stringStart;
                    nextActive.stringIndex = nActiveStrings+1;
                end
            end
        end
    end

    % check the sufficient condition
    normSqrd = 0;
    totalGapOff = 0;
    totalGapInact = 0;

    for iGroup_ = 1:nGroups
        thisPartition = partition(iGroup_);
        inactiveIndices = thisPartition.getindices('inactive');
        inactiveGradient = currentGradient(inactiveIndices);

        normSqrd = normSqrd + thisPartition.getnormone^2;

        if isempty(thisPartition.getindices('active'))
            % Conjecture: this should never fail, if it does take it out, not proven.
            assert(all(inactiveGradient == 0));
            continue;
        end
        if isempty(inactiveIndices)
            continue;
        end

        tmpMax = max(abs(inactiveGradient));
        if ~isempty(thisPartition.getindices('active'))
            totalGapOff = totalGapOff + tmpMax^2;
        else
            totalGapInact = totalGapInact + tmpMax^2;
        end
    end
    passed = (totalGapOff - mu^2 * normSqrd <= 2 * mu * DUALITYGAP) ...
        && (totalGapInact - mu^2 * normSqrd <= 2 * mu * DUALITYGAP);
    end


    function [tmpMax, tmpIx] = expandStringS(first, last)
    % Check towards which end the sequence should expand (for sufficient condition)

    % CAREFUL! This is a nested function and uses the same variables as the main one!

    extIndices = [first-1, last+1];
    tmpIx = extIndices((extIndices >= 1) & (extIndices <= nFeatures));
    tmpIx = tmpIx(:);

    tmpMax = nan(length(tmpIx), 1);

    for ii = 1:length(tmpIx)
        thisIndex_ = tmpIx(ii);
        extGroup = featureGroup(thisIndex_);
        extPartition = partition(extGroup);
        inactiveGradient = currentGradient(thisIndex_);

        if isempty(extPartition.getindices('active'))
            assert(inactiveGradient == 0);
            continue;
        end

        extMargin = inactiveGradient^2;
        tmpMax(ii) = extMargin;
    end
    end


% end of nested functions
end


function [newList, newNActive] = compactStrings(oldList, oldNActive)
% Go throuhg the string list and join all consecutive strings.

newList = oldList;
newNActive = oldNActive;

ii = 0;
while ii < newNActive-1
    ii = ii + 1;
    jj = ii;
    while jj < newNActive
        jj = jj + 1;
        compact = false;
        if newList(ii).first == newList(jj).last + 1
            first = newList(jj).first;
            last = newList(ii).last;
            compact = true;
        elseif newList(ii).last == newList(jj).first - 1
            first = newList(ii).first;
            last = newList(jj).last;
            compact = true;
        end
        if compact
            newList(ii).first = first;
            newList(ii).last = last;
            newList(jj:end-1) = newList(jj+1:end);
            newList(end).first = [];
            newList(end).last = [];
            newNActive = newNActive - 1;
            jj = jj - 1;
        end
    end
end
end
