This is an overview of some benchmark results to compare the performance
of different operator architectures on various problems.

The benchmarks are implemented in the `benchmarks` directory and we refer to
this directory for detailed information on how the benchmarks are run.

## [NavierStokes](../api/continuity/benchmarks/#continuity.benchmarks.NavierStokes)

Reference: _Li, Zongyi, et al. "Fourier neural operator for parametric partial
differential equations." arXiv preprint arXiv:2010.08895 (2020)_ _Table 1 ($\nu$ = 1e−5  T=20  N=1000)_

_reported for_ FNO-3D: __0.1893__ (rel. test error)

[FourierNeuralOperator](../api/continuity/operators/#continuity.operators.FourierNeuralOperator):
0.0185 (rel. train error)  __0.1841__ (rel. test error)

<table>
<tr>
<td>
Best training sample<br>
<img src="img/ns_train_237.png" alt="Best training sample"/>
rel. error = 8.8748e-03
</td>
<td>
Worst training sample<br>
<img src="img/ns_train_420.png" alt="Worst training sample"/>
rel. error = 3.1433e-02
</td>
<td>
Best test sample<br>
<img src="img/ns_test_144.png" alt="Best test sample"/>
rel. error = 1.0220e-01
</td>
<td>
Worst test sample<br>
<img src="img/ns_test_179.png" alt="Worst test sample"/>
rel. error = 4.4655e-01
</td>
</tr>
</table>

{% include 'benchmarks/table.html' %}
