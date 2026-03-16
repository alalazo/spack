% Copyright Spack Project Developers. See COPYRIGHT file for details.
%
% SPDX-License-Identifier: (Apache-2.0 OR MIT)

%=============================================================================
% Display Results
%
% This section determines what parts of the model are printed at the end
%=============================================================================

% Spec attributes
#show attr/2.
#show attr/3.
#show attr/4.
#show attr/5.
#show attr/6.
% names of optimization criteria
#show opt_criterion/2.

% error types
#show error/3.

% for error causation (used by Python-side cause reconstruction)
#show condition_reason/2.
#show condition_holds/2.
