# DARKolo is Where Econ-ARK and Dolo/DolARK Connect

The long run plan for Econ-ARK/HARK is to have all of our models described by a language that will be a descendant of Dolo.  The working name for the prototype language is DolARK (though that could change).

The purpose is NOT to insist that all our models be *solved* or *simulated* by Dolo/DolARK.  (Though when that is possible it will be nice).

Rather, the purpose is to have each model described with a sufficient degree of completeness and clarity that there is no ambiguity about 'what the model is.'

So, for example, if we describe a shock in a model as being distributed according to a lognormal distribution truncated at two standard deviations, then any specific computational solution to the model can have its degree of imperfection measured by the difference between the solution actually obtained, and the solution that would be obtained with an arbitrary degree of precision.  

This serves a number of goals, the most important of these is to provide an unambiguous answer to a plaguing question in computational modeling.  When two teams of people claim to have solved 'the same model' but get numerically different results, how is it possible to judge which is the better solution?  The answer is that the better solution is the one that comes closer to the solution that would be obtained using a computer with an arbitrarily large amount of computational power.  In the example above, if the discrepancies turned out to be caused by the fact that one team was using a 3 point equiprobable approximation to the lognormal distribution, while the other team was using a 21 point Gauss-Hermite approximation, it would be clear that the second team's answer is closer to right.

Categories of content in the repo:

## Chimeras : Solutions of the Same Model in DolARK and in HARK

The requirements for running chimeras are the union of the requirements for Econ-ARK and dolARK (see the /binder directory)

To start with, we aim to have the following chimeras:

1. [BufferStockTheory](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory)
1. [KrusellSmith]()
1. [Aiyagari]()
1. [cstwMPC]()
1. [qModel]()

