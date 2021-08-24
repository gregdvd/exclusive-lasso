classdef Subset < handle
%SUBSET Construct a Subset object
%   SS = SUBSET(ELEMENTS, INDICES) constructs a subset containing the reals
%   in ELEMENTS, which correspond to the elements in positions INDICES in the
%   original vector.
%
%   SUBSET properties:
%       nActive (private)       - number of active elements
%       elements (private)      - vector of elements (ordered absolute value)
%       indices (private)       - positions in the original vector
%       activeMask (private)    - true if element is active
%
%   SUBSET methods:
%       SUBSET/SUBSET   - Construct a SUBSET object
%       activateIndices - Activate given indices
%       activateNext    - Activate highest inactive element
%       checksolution   - Check the necessary condition for the Lagrange multiplier
%       getindices      - Return the indices associated with this subset
%       getnactive      - Return the number of active elements
%       getnextbound    - Return the upperbound resulting from next activation
%       getnormone      - Return the l1 norm of the active elements
%
%   See also Subset.activateNext, Subset.checksolution, Subset.getnactive,
%   Subset.getnextbound, Subset.getnormone.

    properties (Access = private)
        nActive
        elements
        indices
        activeMask
    end

    methods
        function obj = Subset(allElements, newIndices)
            if nargin > 0
                tmp = allElements(newIndices);
                [~, ix] = sort(abs(tmp), 'descend');
                obj.elements = reshape(tmp(ix), [], 1);
                obj.indices = reshape(newIndices(ix), [], 1);
                obj.activeMask = false(length(newIndices), 1);
                obj.nActive = 0;
            end
        end

        function s = activateIndices(obj, indices)
        %ACTIVATEINDICES activate given indices.
        %   s = ACTIVATEINDICES(obj, indices) activates the indices specified as
        %   input.

            indices = indices(:); % ensure column
            [mask, positions] = ismember(indices, obj.indices);
            if ~all(mask)
                error('Not a valid set of indices for this Subset.');
            end

            s = sum(mask);
            obj.nActive = obj.nActive + sum(xor(obj.activeMask(positions), true(s, 1)));
            obj.activeMask(positions) = obj.activeMask(positions) | true(s, 1);
        end

        function indices = activateNext(obj)
        %ACTIVATENEXT activate highest unused element.
        %   ix = ACTIVATENEXT activates the highest (in magnitude) unused element and
        %   returns the corresponding index in the original vector. If there exist
        %   more than one element with the same magnitude, all of them are activated
        %   and all indices are returned.

            if obj.nActive == length(obj.elements) || all(obj.elements == 0)
                indices = [];
            else
                obj.nActive = obj.nActive + 1;
                obj.activeMask(obj.nActive) = true;
                nSelected = 1;
                selectedIndices = nan(sum(~obj.activeMask), 1);
                selectedIndices(1) = obj.indices(obj.nActive);
                while obj.nActive < length(obj.elements)...
                        && abs(obj.elements(obj.nActive)) == abs(obj.elements(obj.nActive + 1))
                    obj.nActive = obj.nActive + 1;
                    nSelected = nSelected + 1;
                    selectedIndices(nSelected) = obj.indices(obj.nActive);
                    obj.activeMask(obj.nActive) = true;
                end
                indices = selectedIndices(1:nSelected);
            end
        end

        function indices = getindices(obj, indexType)
        %GETINDICES return the indices associated to this Subset.

            if nargin == 1
                indexType = 'all';
            end

            nElements = length(obj.indices);
            switch indexType
                case 'all'
                    mask = true(nElements, 1);
                case 'active'
                    mask = obj.activeMask;
                case 'inactive'
                    mask = ~obj.activeMask;
                otherwise
                    error('Unkown index type.');
            end
            indices = obj.indices(mask);
        end

        function y = getnormone(obj)
        %GETNORMONE compute the l1 norm of the active elements.

            y = sum(abs(obj.elements(obj.activeMask)));
        end

        function n = getnactive(obj)
        %GETNACTIVE return the number of active elements.

            n = sum(obj.activeMask);
        end

        function [bound, ix] = getnextbound(obj)
        %GETNEXTBOUND compute the upperbound resulting from activating the next element
        %   [bound, ix] = GETNEXTBOUND computes the upperbound value that will result
        %   after activating the next element(s). ix contains the indices (in the
        %   original vector) of the next elements to be activated.

            if obj.nActive < length(obj.elements)
                mask = (obj.elements == obj.elements(obj.nActive + 1));
                ix = obj.indices(mask);
                lots = obj.nActive + sum(mask);
                if lots < length(obj.elements)
                    bound = sum(abs(obj.elements(1:lots))) / abs(obj.elements(lots + 1)) - lots;
                else
                    bound = inf;
                end
            else
                bound = inf;
                ix = [];
                return
            end
        end

        function [s, l, u] = checksolution(obj, mu)
        %CHECKSOLUTION Check the necessary condition for the object group.
        %   s = CHECKSOLUTION(SS, mu) is true if multiplier mu satisfies the necessary
        %   condition for the Subset object SS.
        %   [s, l, u] = CHECKSOLUTION(SS, mu) also returns the difference between mu
        %   and the lowerbound (l) and between mu and the upperbound (u). l and u take
        %   a negative value if the corresponding constraint is not met.

            lowerbound = sum(abs(obj.elements(1:obj.nActive-1)))...
                / abs(obj.elements(obj.nActive)) - obj.nActive + 1;
            if obj.nActive < length(obj.elements)
                upperbound = sum(abs(obj.elements(1:obj.nActive)))...
                    / abs(obj.elements(obj.nActive+1)) - obj.nActive;
            else
                upperbound = inf;
            end

            s = (mu >= lowerbound) && (mu < upperbound);

            if nargout > 1
                l = mu - lowerbound;
                u = upperbound - mu;
            end
        end

    end
end
