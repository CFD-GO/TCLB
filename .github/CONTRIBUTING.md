## Things we would appreciate:

We are really glad that you would like to contribute to TCLB. We would appreciate if you first take a little time to read the points below before you proceed.

### Code references
- [ ] Original references - if you implement a model/feature from some paper/book, please leave a reference in the code. **NOTICE:** This is very important, and your pull request may be rejected or postponed if you are not able to provide reference information. If you are the author of the model (not only the implementation), please also leave some references or preferred citation
- [ ] Wiki references - if you added feature or configure arguments, please add it as a wiki entry and reference it in the code
- [ ] Author references - if you added a model, complicated handler or physics based feature, please provide the details of the implementation author as: `Maintainer: your-name @your-github-login`

### Comments in the pull request
- [ ] Issues references - if your pull request is solving/changing some issue, please reference it in comments (fixes #NUMBER) or name of this pull request

### Coding standards
- [ ] Please don't break naming conventions (at least look at surrounding code)
- [ ] Try to use meaningful variable names
- [ ] Do not pull/merge XXX_new's.  It's either something different and should have a different name, or it is a replacement of XXX (in whole code)

### [Pirate's Code aka Guidelines](https://www.youtube.com/watch?v=6GMkuPiIZ2k)
- [ ] A lot of classes and useful functions are already implemented. A least try to find these. And yes, we know that they should be in the dev's wiki. We will work on that some day.
- [ ] You found a clever trick? Please leave a suitable comment

### Compilation
The code will be tested automatically with Travis-CI, but before that, you can test your code on your machine:
- [ ] Paranoid CPU build: `./configure --disable-cuda --enable-paranoid`
- [ ] CUDA capable GPU build (capability 2.0): `./configure --with-cuda-arch=sm_20`

You can also activate Travis automatic builds on your fork of the code.
