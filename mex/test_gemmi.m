classdef test_gemmi < matlab.unittest.TestCase
    %TEST_GEMMI Unit tests for the gemmi MEX interface.
    %
    %   Run from the workspace root (or any directory on the MATLAB path
    %   that contains the compiled gemmi MEX file) with:
    %
    %       results = runtests('mex/test_gemmi.m')
    %       table(results)

    properties (Constant)
        % Relative Frobenius-norm tolerances.
        TolDouble = 1e-14;
        TolSingle = 1e-6;
        % Canonical default option struct.
        DefaultOpts = struct('split', 'n', 'mult', 'r', 'acc', 'i')
    end

    %%%%%%%%%%%%%%%%%%
    % Helper methods %
    %%%%%%%%%%%%%%%%%%

    % Setup: reset persistent MEX options to defaults, so that tests are
    % independent of the execution order.
    methods (TestMethodSetup)
        function resetToDefaults(tc)
            A = ones(2, 2, 'double');
            B = ones(2, 2, 'double');
            gemmi(A, B, 10, 10, tc.DefaultOpts);
        end
    end

    methods (Access = private)
        function relErr = relFrobErr(~, C, Cref)
            %RELFROB Relative Frobenius-norm error ||C - Cref||_F / ||Cref||_F.
            relErr = norm(C - Cref, 'fro') / norm(Cref, 'fro');
        end

        function verifyAlgout(tc, algout, expected, context)
            %VERIFYALGOUT Check that ALGOUT has the expected fields and values.
            tc.verifyTrue(isfield(algout, 'split'), sprintf('algout missing field "split" (%s)', context));
            tc.verifyTrue(isfield(algout, 'mult'), sprintf('algout missing field "mult" (%s)', context));
            tc.verifyTrue(isfield(algout, 'acc'), sprintf('algout missing field "acc" (%s)', context));
            tc.verifyEqual(algout.split, expected.split, sprintf('Unexpected split value (%s)', context));
            tc.verifyEqual(algout.mult, expected.mult, sprintf('Unexpected mult value (%s)', context));
            tc.verifyEqual(algout.acc, expected.acc, sprintf('Unexpected acc value (%s)', context));
        end

        function assertThrowsContaining(tc, fn, msgFragment)
            %ASSERTTHROWSCONTAINING Verify fn() throws and exception message
            %  contains msgFragment.
            caught = false;
            try
                fn();
            catch e
                caught = true;
                tc.assertTrue( ...
                    contains(e.message, msgFragment), ...
                    sprintf('Expected message containing "%s", got: "%s"', ...
                        msgFragment, e.message));
            end
            tc.assertTrue(caught, ...
                sprintf('Expected an error containing "%s" but none was thrown.', ...
                    msgFragment));
        end

        function assertErrorCases(tc, cases)
            %ASSERTERRORCASES Run a table of invalid calls and expected message
            %  fragments.
            for i = 1:size(cases, 1)
                tc.assertThrowsContaining(cases{i, 1}, cases{i, 2});
            end
        end
    end

    %%%%%%%%%
    % Tests %
    %%%%%%%%%

    methods (Test)

        function testSuccessfulComputations(tc)

            % Test a variety of shapes and data types, with default options.
            cases = {
                'double-square',   @() randn(10, 10),              @() randn(10, 10),              tc.TolDouble, 'double';
                'double-nonsquare',@() randn(5, 12),               @() randn(12, 7),               tc.TolDouble, 'double';
                'single-square',   @() randn(10, 10, 'single'),    @() randn(10, 10, 'single'),    tc.TolSingle, 'single';
                'single-nonsquare',@() randn(5, 12, 'single'),     @() randn(12, 7, 'single'),     tc.TolSingle, 'single';
            };

            numSlices = 15;

            for i = 1:size(cases, 1)
                label = cases{i, 1};
                A = cases{i, 2}();
                B = cases{i, 3}();
                tol = cases{i, 4};
                expectedClass = cases{i, 5};

                C = gemmi(A, B, numSlices);
                relErr = tc.relFrobErr(C, A * B);
                if isa(C, 'single')
                    relErr = double(relErr);
                end

                tc.verifyLessThan(relErr, tol, sprintf('Relative error too large for case: %s', label));
                tc.verifyClass(C, expectedClass, sprintf('Wrong output class for case: %s', label));
                tc.verifySize(C, [size(A, 1), size(B, 2)], sprintf('Wrong output shape for case: %s', label));
            end

            % Test special matrices.
            n = 5;
            rng(5);
            A = randn(n, n);
            B = randn(n, n);

            % Test identity matrix.
            tc.verifyLessThan(tc.relFrobErr(gemmi(A, eye(n), 10), A), tc.TolDouble, 'Right multiplication by the identity should preserve A.');
            tc.verifyLessThan(tc.relFrobErr(gemmi(eye(n), B, 10), B), tc.TolDouble, 'Left multiplication by the identity should preserve B.');

            % Test zero matrix.
            tc.verifyEqual(gemmi(A, zeros(n, n), 10), zeros(n, n), 'Right multiplication by zero should return zero.');
            tc.verifyEqual(gemmi(zeros(n, n), A, 10), zeros(n, n), 'Left multiplication by zero should return zero.');

            % Test number of slices.
            m = 6; k = 8; n = 4;
            A = randn(m, k);
            B = randn(k, n);
            tc.verifyEqual(gemmi(A, B, numSlices), gemmi(A, B, numSlices, numSlices),...
                'Three-argument and symmetric four-argument calls should match.');
            tc.verifyLessThan(tc.relFrobErr(gemmi(A, B, numSlices, 2 * numSlices), A * B), tc.TolDouble,...
                'Asymmetric split counts should still be accurate.');

            % Test non-default algorithmic variants.
            cases = {
                'split-t', struct('split', 't');
                'split-u', struct('split', 'u');
                'mult-f',  struct('mult', 'f');
                'acc-f',   struct('acc', 'f');
            };
            for i = 1:size(cases, 1)
                label = cases{i, 1};
                opts = cases{i, 2};
                rng(9 + i);
                A = randn(8, 8);
                B = randn(8, 8);
                C = gemmi(A, B, 10, 10, opts);
                tc.verifyLessThan(tc.relFrobErr(C, A * B), tc.TolDouble, ...
                    sprintf('Accuracy failed for variant: %s', label));
            end
        end

        function testAlgout(tc)

            % Test that ALGOUT is returned correctly for default values.
            n = 5;
            rng(5);
            A = randn(n, n);
            B = randn(n, n);
            tc.verifyClass(gemmi(A, B, 10), 'double', 'Single-output default call should return double output.');

            [C, algout] = gemmi(A, B, 10);
            tc.verifyClass(C, 'double', 'Two-output default call should return double output.');
            tc.verifyAlgout(algout, tc.DefaultOpts, 'two-output default call');

            [~, algout] = gemmi(A, B, 10, 10, tc.DefaultOpts);
            tc.verifyAlgout(algout, tc.DefaultOpts, 'explicit default options');

            % Test correctness of ALGOUT fields.
            cases = {
                'split', 't';
                'split', 'u';
                'split', 'n';
                'mult',  'f';
                'mult',  'r';
                'acc',   'f';
                'acc',   'i';
            };
            for i = 1:size(cases, 1)
                field = cases{i, 1};
                value = cases{i, 2};
                [~, algout] = gemmi(A, B, 10, 10, struct(field, value));
                tc.verifyEqual(algout.(field), value, ...
                    sprintf('ALGOUT mismatch for %s=%s', field, value));
            end

            % Test persistence of options across calls.
            gemmi(A, B, 10, 10, struct('split', 't'));
            [~, algout] = gemmi(A, B, 10);
            tc.verifyAlgout(algout, struct('split', 't', 'mult', 'r', 'acc', 'i'), ...
                'persisted options after split=t');

            [~, algout] = gemmi(A, B, 10, 10, tc.DefaultOpts);
            tc.verifyAlgout(algout, tc.DefaultOpts, 'reset to explicit defaults');
        end

        function testInvalidCalls(tc)
            tc.assertErrorCases({
                @() gemmi(ones(2,2), ones(2,2)), 'Three to five inputs expected';
                @() gemmi(ones(2,2), ones(2,2), 10, 10, struct(), struct()), 'Three to five inputs expected';
                @() gemmi(ones(3, 4), ones(5, 3), 10), 'conformable';
                @() gemmi(ones(3,3) + 1i*ones(3,3), ones(3,3), 10), 'real matrix';
                @() gemmi(ones(3,3), ones(3,3) + 1i*ones(3,3), 10), 'real matrix';
                @() gemmi(ones(3,3,'double'), ones(3,3,'single'), 10), 'same data type';
                @() gemmi(ones(2,2,2), ones(4,3), 10), 'matrices';
                @() gemmi(ones(3,4), ones(2,2,2), 10), 'matrices';
                @() gemmi(ones(3,3), ones(3,3), 1.5), 'scalar integer';
                @() gemmi(ones(3,3), ones(3,3), 10, 2.7), 'scalar integer';
                @() gemmi(ones(3,3), ones(3,3), 10, 10, 'invalid'), 'struct';
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('split','n', 'mult','r', 'acc','i', 'extra','x')), 'at most three fields';
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('badfield', 'n')), "named 'split', 'mult', or 'acc'";
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('split', 'nn')), 'single character';
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('split','x')), "split' is invalid";
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('mult','x')), "mult' is invalid";
                @() gemmi(ones(3,3), ones(3,3), 10, 10, struct('acc','x')), "acc' is invalid";
            });

            caught = false;
            try
                [~, ~, ~] = gemmi(ones(2,2), ones(2,2), 10);
            catch e
                caught = true;
                tc.assertTrue( ...
                    contains(e.message, 'at most two output arguments'), ...
                    sprintf('Unexpected message: "%s"', e.message));
            end
            tc.assertTrue(caught, 'Expected an error but none was thrown');
        end

    end
end
