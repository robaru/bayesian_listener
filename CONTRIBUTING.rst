.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. The following helps you to start
contributing specifically to bayesian_listener.

Types of Contributions
----------------------

Report Bugs or Suggest Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best place for this is https://github.com/robaru/bayesian_listener_package/issues.

Fix Bugs or Implement Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Look through https://github.com/robaru/bayesian_listener_package/issues for bugs or feature request
and contact us or comment if you are interested in implementing.

Write Documentation
~~~~~~~~~~~~~~~~~~~

bayesian_listener could always use more documentation, whether as part of the
official bayesian_listener docs, in docstrings, or even on the web in blog posts,
articles, and such.

If you are planning to show code in the documentation, add a corresponding
test in ``tests/test_guide_<guide_name>.py``.  Mark it so it is skipped in
the default test run but can be executed on demand::

    import pytest
    pytestmark = pytest.mark.guide

    def test_my_example(sofa_path):
        # [my_section]
        from bayesian_listener import BayesianListener
        bl = BayesianListener(sofa_path)
        estimates = bl.localise(repetitions=1, seed=0)
        # [/my_section]
        assert estimates is not None

The ``# [my_section]`` / ``# [/my_section]`` markers let Sphinx pull the
snippet into the docs via ``literalinclude`` with ``start-after`` /
``end-before``.  Run the guide tests with::

    $ pytest -m guide

Get Started!
------------

Ready to contribute? Here's how to set up `bayesian_listener_package` for local development using the command-line interface. Note that several alternative user interfaces exist, e.g., the Git GUI, `GitHub Desktop <https://desktop.github.com/>`_, extensions in `Visual Studio Code <https://code.visualstudio.com/>`_ ...

1. `Fork <https://docs.github.com/en/get-started/quickstart/fork-a-repo/>`_ the `bayesian_listener` repo on GitHub.
2. Clone your fork locally and cd into the bayesian_listener_package directory::

    $ git clone https://github.com/robaru/bayesian_listener_package.git
    $ cd bayesian_listener_package

3. Install your local copy into a virtualenv. Assuming you have Anaconda or Miniconda installed, this is how you set up your fork for local development::

    $ conda create --name bayesian_listener_package python
    $ conda activate bayesian_listener_package
    $ pip install -e ".[dev]"

4. Create a branch for local development. Indicate the intention of your branch in its respective name (i.e. `feature/branch-name` or `bugfix/branch-name`)::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass ruff and the
   tests::

    $ ruff check
    $ pytest

   ruff must pass without any warnings for `./bayesian_listener` and `./tests` using the default or a stricter configuration. Ruff ignores a couple of PEP Errors (see `./pyproject.toml`). If necessary, adjust your linting configuration in your IDE accordingly.

6. If you changed docstrings or ``.rst`` files, build the docs locally to check
   for warnings::

    $ python -m sphinx -W --keep-going -b html docs docs/_build/html

   This mirrors the CI build (``-W`` promotes warnings to errors). For a
   quicker preview without strict warnings::

    $ python -m sphinx -b html docs docs/_build/html

   Then open ``docs/_build/html/index.html`` in a browser.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request on the develop branch through the GitHub website.



Keep your branch up to date with rebase
---------------------------------------
Before submitting your pull request, make sure your branch is up to date with
the latest ``develop`` branch using rebase rather than merge. This keeps the
commit history clean and linear::

    $ git fetch origin
    $ git rebase origin/develop

If conflicts arise, Git will pause and indicate the conflicting files. For each
conflict:

1. Open the file and resolve the conflict manually.
2. Stage the resolved file::

    $ git add <conflicted-file>

3. Continue the rebase::

    $ git rebase --continue

Repeat until all conflicts are resolved. If at any point you want to start
over::

    $ git rebase --abort

Once the rebase is complete, push your updated branch. Since rebase rewrites
commit history, a force push is required::

    $ git push --force-with-lease origin name-of-your-bugfix-or-feature

.. note::
    Use ``--force-with-lease`` instead of ``--force``. It is a safer option
    that prevents overwriting changes if someone else has pushed to the same
    branch in the meantime.

Releasing a new version
-----------------------

The release procedure follows the
`pyfar release guidelines <https://pyfar-gallery.readthedocs.io/en/latest/contribute/packages/releasing.html>`_,
with two differences: branches are rebased rather than merged, and every
release goes through a pull request into ``main`` so that CI tests the exact
code that will be published. Only maintainers with push access to ``main``
and ``develop`` can release.

Versioning
~~~~~~~~~~

bayesian_listener uses `Semantic Versioning <https://semver.org/>`_,
``MAJOR.MINOR.PATCH``:

* ``MAJOR`` – incompatible API changes,
* ``MINOR`` – backwards-compatible new functionality,
* ``PATCH`` – backwards-compatible bug fixes. Documentation-only changes are
  also released as a patch.

Because the model output is used in publications, treat any change that alters
simulated responses, fitted parameters, or metric values for the same input as
a ``MINOR`` change at least, and document it in ``HISTORY.rst``.

Branching
~~~~~~~~~

* ``MINOR`` and ``MAJOR`` releases collect the commits on ``develop``. The
  release branch is ``develop`` itself, rebased onto ``main``.
* ``PATCH`` releases are prepared on a ``bugfix/<name>`` branch created from
  ``main``.

In both cases the release branch is brought into ``main`` with a pull request
that is merged with the *rebase* strategy, never with a merge commit. ``main``
is never modified locally. It therefore always has a linear history and every
commit on it has passed CI.

Procedure
~~~~~~~~~

1. Bring the release branch up to date with ``main`` by rebasing. For a
   minor or major release::

    $ git fetch origin
    $ git checkout develop
    $ git rebase origin/main

   For a patch release::

    $ git fetch origin
    $ git checkout -b bugfix/<name> origin/main

   Resolve conflicts as described in `Keep your branch up to date with
   rebase`_.

2. Update ``HISTORY.rst`` on the release branch. Add a new section at the
   top with the version number and release date, grouped into ``Added``,
   ``Changed``, ``Fixed`` and ``Removed`` sub-sections as applicable::

    0.2.0 (2026-09-18)
    ------------------

    Added
    ^^^^^
    * New feature ...

    Fixed
    ^^^^^
    * Bug fix ...

   Commit the changelog::

    $ git add HISTORY.rst
    $ git commit -m "docs: update HISTORY.rst for 0.2.0"

3. Run the full test suite, the linter and the strict documentation build
   locally, mirroring CI. All must pass without warnings::

    $ pytest
    $ pytest tests -W error::DeprecationWarning
    $ ruff check
    $ python -m sphinx -W --keep-going -b html docs docs/_build/html

4. Push the release branch and open a pull request against ``main``. Because
   the rebase rewrote history, ``develop`` needs a force push::

    $ git push --force-with-lease origin develop
    $ gh pr create --base main --head develop --title "Release 0.2.0"

   For a patch release push ``bugfix/<name>`` normally. The pull request can
   also be opened on the GitHub website. CircleCI runs the tests, ruff, the
   documentation build and the deprecation-warning run on the pushed branch
   and reports the result on the pull request.

5. Once CI is green and the pull request is approved, merge it with the
   rebase strategy so ``main`` stays linear. Never merge into ``main``
   locally::

    $ gh pr checks --watch
    $ gh pr merge --rebase

   The same can be done on the GitHub website with *Rebase and merge*. If
   GitHub reports that the branch is not up to date, ``main`` moved in the
   meantime: go back to step 1.

6. Check out the merged ``main`` and bump the version. The
   ``bump-my-version`` configuration in ``pyproject.toml`` updates the
   version in ``pyproject.toml`` and ``bayesian_listener/__init__.py``,
   creates a commit and a ``vX.Y.Z`` tag. The working tree must be clean::

    $ git checkout main
    $ git pull origin main
    $ bump-my-version bump minor --verbose

   Replace ``minor`` with ``patch`` or ``major`` as appropriate. Use
   ``--dry-run`` first to preview the changes.

7. Push the commit and the tag::

    $ git push --follow-tags

   Pushing a ``vX.Y.Z`` tag triggers the ``test_and_publish`` workflow on
   CircleCI, which re-runs all checks on the tagged commit. If all jobs pass,
   the package is built and uploaded to PyPI automatically. Check the
   workflow on CircleCI and the new version on
   https://pypi.org/project/bayesian_listener/.

8. Bring ``develop`` level with ``main`` so the version bump is carried
   over. The rebase merge on GitHub rewrote the commit hashes, so rebase
   ``develop`` onto ``main``; git drops the commits that are already there::

    $ git checkout develop
    $ git rebase main
    $ git push --force-with-lease origin develop

9. Create a GitHub release from the tag and copy the ``HISTORY.rst`` entry
   into its description. Acknowledge contributors where appropriate.

.. note::
    If the ``test_and_publish`` workflow fails after the tag was pushed, fix
    the problem through a new pull request, delete the tag locally and
    remotely (``git tag -d vX.Y.Z`` and ``git push origin :refs/tags/vX.Y.Z``),
    revert the bump commit and repeat from step 6. A version that was already
    uploaded to PyPI cannot be replaced; release a new patch version instead.
