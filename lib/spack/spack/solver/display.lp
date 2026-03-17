% Copyright Spack Project Developers. See COPYRIGHT file for details.
%
% SPDX-License-Identifier: (Apache-2.0 OR MIT)

%=============================================================================-
% Display Results
%
% This section determines what parts of the model are printed at the end
%==============================================================================

% Spec attributes
#show attr/2.
#show attr/3.
#show attr/4.
#show attr/5.
#show attr/6.
% names of optimization criteria
#show opt_criterion/2.

% error types
#show error/2.
#show error/3.
#show error/4.
#show error/5.
#show error/6.


% Serialize from clingo to Python these facts only if we are in presence of an error
% This is to avoid serializing a huge number of needless facts in the happy path, which
% can take a significant amount of time.

has_error :- error(A1, A2).
has_error :- error(A1, A2, A3).
has_error :- error(A1, A2, A3, A4).
has_error :- error(A1, A2, A3, A4, A5).
has_error :- error(A1, A2, A3, A4, A5, A6).

% for error causation
#show condition_reason(A1, A2) : condition_reason(A1, A2), has_error.

% For error messages to use later
#show pkg_fact(A1, A2)        : pkg_fact(A1, A2), has_error.
#show condition_holds(A1, A2) : condition_holds(A1, A2), has_error.

#show imposed_constraint(A1, A2, A3)             : imposed_constraint(A1, A2, A3), has_error.
#show imposed_constraint(A1, A2, A3, A4)         : imposed_constraint(A1, A2, A3, A4), has_error.
#show imposed_constraint(A1, A2, A3, A4, A5)     : imposed_constraint(A1, A2, A3, A4, A5), has_error.
#show imposed_constraint(A1, A2, A3, A4, A5, A6) : imposed_constraint(A1, A2, A3, A4, A5, A6), has_error.

#show condition_requirement(A1, A2, A3)             : condition_requirement(A1, A2, A3), has_error.
#show condition_requirement(A1, A2, A3, A4)         : condition_requirement(A1, A2, A3, A4), has_error.
#show condition_requirement(A1, A2, A3, A4, A5)     : condition_requirement(A1, A2, A3, A4, A5), has_error.
#show condition_requirement(A1, A2, A3, A4, A5, A6) : condition_requirement(A1, A2, A3, A4, A5, A6), has_error.

#show node_has_variant(A1, A2, A3)   : node_has_variant(A1, A2, A3), has_error.
#show trigger_and_effect(A1, A2, A3) : trigger_and_effect(A1, A2, A3), has_error.
#show trigger_node(A1, A2, A3)       : trigger_node(A1, A2, A3), has_error.

#show build(A1)    : build(A1), has_error.
#show external(A1) : external(A1), has_error.

#show unification_set(A1, A2)      : unification_set(A1, A2), has_error.
#show provider(A1, A2)             : provider(A1, A2), has_error.
#show condition_nodes(A1, A2)      : condition_nodes(A1, A2), has_error.
#show imposed_nodes(A1, A2)        : imposed_nodes(A1, A2), has_error.
#show variant_single_value(A1, A2) : variant_single_value(A1, A2), has_error.

% debug
