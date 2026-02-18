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

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request on the develop branch through the GitHub website.



Keep your branch up to date with rebase
------------
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

