import numpy as np
import scipy.sparse
import csdl_alpha as csdl


def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)
    knots = np.zeros(num_cp + order)
    knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1:] = 1.0
    t_vec = np.linspace(0, 1, num_pt)
    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)
    for ipt in range(num_pt):
        t = t_vec[ipt]
        i0 = -1
        for ind in range(order, num_cp + 1):
            if (knots[ind - 1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order
        basis[:] = 0.
        basis[-1] = 1.
        for i in range(2, order + 1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n + l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.
            for j in range(j1 + 1, j2):
                n = i0 + j
                if knots[n + l - 1] != knots[n - 1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j - 1] = 0.
                if knots[n + l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n + l - 1] != knots[n - 1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2 - 1] = 0.
        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange
    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_pt, num_cp),
    )


class BsplineComp(csdl.CustomExplicitOperation):
    """
    Translates control points to actual points using a B-spline.
    """
    def __init__(self, num_cp, num_pt, jac, in_name, out_name):
        super().__init__()
        self.num_cp = num_cp
        self.num_pt = num_pt
        self.jac = jac
        self.in_name = in_name
        self.out_name = out_name

    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input(self.in_name, getattr(inputs, self.in_name))
        output = self.create_output(self.out_name, shape=(self.num_pt,))
        self.declare_derivative_parameters(self.out_name, self.in_name, dependent=True)
        out = csdl.VariableGroup()
        setattr(out, self.out_name, output)
        return out

    def compute(self, input_vals, output_vals):
        output_vals[self.out_name] = self.jac @ input_vals[self.in_name]

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives[self.out_name, self.in_name] = self.jac.toarray().astype(float)


if __name__ == "__main__":
    num_cp = 10
    num_pt = 100

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    inputs = csdl.VariableGroup()
    inputs.cp = csdl.Variable(value=np.random.rand(num_cp), name='cp')
    inputs.cp.set_as_design_variable()

    jac = get_bspline_mtx(num_cp, num_pt)
    op = BsplineComp(num_cp=num_cp, num_pt=num_pt, jac=jac,
                     in_name='cp', out_name='pt')
    outputs = op.evaluate(inputs)
    pt = outputs.pt
    pt_sum = csdl.sum(pt)
    pt_sum.set_as_objective()
    pt_sum.add_name('pt_sum')

    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder=recorder)
    sim.run()
    # sim.check_totals()
    sim.compute_totals(pt_sum, inputs.cp)

    print("B-spline output shape:", pt.value.shape)
    print("First 5 values:", pt.value[:5])