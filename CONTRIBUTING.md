<!--
SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.

SPDX-License-Identifier: Apache-2.0
-->

# How to contribute

We'd love to accept your patches and contributions to this project. There are just a few guidelines you need to follow.

## Ways of contributing

Contribution does not necessarily mean committing code to the repository.
We recognize different levels of contributions as shown below in increasing order of dedication.

1. Use and test the project. Give feedback on the user experience or suggest new features.
2. Report bugs or security vulnerabilities.
3. Fix bugs.
4. Improve the project by developing new features.

## Filing bugs, security vulnerabilities or feature requests

You can file bugs against and feature requests for the project via GitHub Issues. Read [GitHub Help](https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/creating-an-issue) for more information on using GitHub Issues.

## Community guidelines

This project follows the following [Code of Conduct](CODE_OF_CONDUCT.md).

## REUSE compliance and source code headers

All the files in the repository need to be [REUSE compliant](https://reuse.software/).
We use the pipeline to automatically check this.
If there are files which are not complying, the pipeline will fail the pull request will be blocked.

This means that every file containing source code must include copyright and license information. This includes any JS/CSS files that you might be serving out to browsers. (This is to help well-intentioned people avoid accidental copying that doesn't comply with the license.)

Apache 2.0 header:

```text
    SPDX-FileCopyrightText: Copyright Contributors to the <YOUR PROJECT NAME> project <YOUR_PROJECT_EMAIL_ADRESS@alliander.com>
    SPDX-License-Identifier: Apache-2.0
```

## Git branching

As this project is not intended for production usage, it applies a simple main/feature model for branching for minimal overhead. New features are developed in a feature branch and directly merged back into main.

## Signing the Developer Certificate of Origin (DCO)

This project uses a Developer Certificate of Origin (DCO) to ensure that each commit was written by the author or that the author has the appropriate rights necessary to contribute the change.
Specifically, we use [Developer Certificate of Origin, Version 1.1](http://developercertificate.org/), which is the same mechanism that the Linux® Kernel and many other communities use to manage code contributions.
The DCO is considered one of the simplest tools for sign-offs from contributors as the representations are meant to be easy to read and indicating signoff is done as a part of the commit message.

This means that each commit must include a DCO which looks like this:

`Signed-off-by: Joe Smith <joe.smith@email.com>`

The project requires that the name used is your real name and the e-mail used is your real e-mail.
Neither anonymous contributors nor those utilizing pseudonyms will be accepted.

There are other great tools out there to manage DCO signoffs for developers to make it much easier to do signoffs:

* Git makes it easy to add this line to your commit messages. Make sure the `user.name` and `user.email` are set in your git configs. Use `-s` or `--signoff` to add the Signed-off-by line to the end of the commit message.
* [GitHub UI automatic signoff capabilities](https://github.blog/changelog/2022-06-08-admins-can-require-sign-off-on-web-based-commits/) for adding the signoff automatically to commits made with the GitHub browser UI. This one can only be activated by the GitHub org or repo admin.
* [GitHub UI automatic signoff capabilities via custom plugin]( https://github.com/scottrigby/dco-gh-ui ) for adding the signoff automatically to commits made with the GitHub browser UI.
* Additionally, it is possible to use shell scripting to automatically apply the sign-off. For an example for bash to be put into a .bashrc file, see [here](https://wiki.lfenergy.org/display/HOME/Contribution+and+Compliance+Guidelines+for+LF+Energy+Foundation+hosted+projects).
* Alternatively, you can add `prepare-commit-msg hook` in .git/hooks directory. [See an example](https://github.com/Samsung/ONE-vscode/wiki/ONE-vscode-Developer's-Certificate-of-Origin).

## Code reviews

All patches and contributions, including patches and contributions by project members, require review by one of the maintainers of the project.
We use GitHub pull requests for this purpose.
See [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Pull request process

Contributions should be submitted as GitHub pull requests. See [Creating a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) if you're unfamiliar with this concept.

Follow this process for a code change and pull request:

1. Fork the repository.
1. Make your change in a feature/description_of_your_change branch.
1. Run the tests.
1. Run the [pre-commit hooks](https://pre-commit.com/) to check for code style and other issues.
1. Submit a pull request.
1. Pull requests will be reviewed by one of the maintainers who may discuss, offer constructive feedback, request changes, or approve the work.
1. Upon receiving the sign-off from one of the maintainers will merge it for you.

## Attribution

This CONTRIBUTING.md is adapted from Google, available at https://github.com/google/new-project/blob/master/docs/contributing.md.
