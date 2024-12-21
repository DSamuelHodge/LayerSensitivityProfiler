import torch
import torch.nn as nn
import numpy as np
import math

class TDVP(nn.Module):
    def __init__(self, in_features, out_features, bond_dim, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dim = min(bond_dim, min(in_features, out_features))
        
        # Initialize core and transform matrices
        self.core_matrix = nn.Parameter(torch.empty(out_features, self.bond_dim))
        self.transform_matrix = nn.Parameter(torch.empty(self.bond_dim, in_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.core_matrix, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.transform_matrix, a=math.sqrt(5))
        
        if device is not None:
            self.to(device)
    
    def forward(self, x):
        """Forward pass using matrix multiplication."""
        # x shape: (..., in_features)
        # First apply transform matrix
        transformed = torch.matmul(x, self.transform_matrix.t())  # (..., bond_dim)
        # Then apply core matrix
        return torch.matmul(transformed, self.core_matrix.t())  # (..., out_features)
    
    def get_effective_weight(self):
        """Get the effective weight matrix."""
        return torch.matmul(self.core_matrix, self.transform_matrix)  # (out_features, in_features)


class TDVP_V2(nn.Module):
    """Layer that converts the image input into an MPS
    d:          local Hilbert space dimension: 2(default)
    D:          bond dimension of the MPO: 2(default)
    out_norm:   Normalization of the output vector: "none", "L1" (default), "L2"
    activation: Activation function applied after the local affine transformation
                Possible options: "none", relu(default), expm, exp
    mode:       Parallel or sequential evaluation of the local hamiltonians: parallel(default), sequential
    eps:        Determines the standard deviation of the initializations: 1e-6(default)
    ti:         Determines if the hamiltonian is translationaly invariant: False (default)
    mask_off:   Disables the mask which ensures that all matrices in the MPO are upper triangular
    cyclic:     If enabled we use the trace instead of the left and right boundary vectors
    """

    def __init__(self, n, din=2, dout=2, D=2, out_norm="L1", activation="relu", mode="parallel", eps=1.0, ti=False, cyclic=False, residual=False, scale=1.0, trainable_A=False):
        super(TDVP_V2, self).__init__()

        # Set attributes
        self.n = n
        self.din = din
        self.dout = dout
        self.D = D
        self.eps = eps
        self.out_norm = out_norm
        self.activation = activation
        self.mode = mode
        self.ti = ti
        self.residual = residual
        self.trainable_A = trainable_A
        self.cyclic = cyclic

        if self.ti:
            logging.info("Using a translationally invariant TDVP.")

        # Activation functions
        self.LeakyReLU = torch.nn.LeakyReLU(negative_slope=1e-5)
        self.sigmoid = torch.nn.Sigmoid()

        self.register_parameter("scale", param=nn.Parameter(
            torch.tensor(scale, dtype=torch.float32)))

        if self.residual:
            self.register_parameter(name='res_coef',
                                    param=nn.Parameter(torch.tensor(1.0, dtype=torch.float32)))

        if self.ti:
            A = np.random.randn(D, D, din)*eps
            if not self.trainable_A:
                if din == 2 and D == 2:
                    A[:2, :, :] = 0
                    A[:, :2, :] = 0
                    A[:2, :2, :] += np.array(
                        [[[1,  1], [1,  0]], [[0,  1], [0,  1]]])
                if din == 2 and D >= 4 and D < 10:
                    A[4:, 4:, :] = np.random.randn(D-4, D-4, din)
                    A[4:, 4:, :] = np.random.randn(D-4, D-4, din)
                    A[:4, :4, 0] = np.array([[1., 1., 0., 0.], [0., 0., 0., 0.], [
                                            1., 1., 0., 0.], [0., 0., 0., 0.]])
                    A[:4, :4, 1] = np.array([[0., 0., 0., 0.], [0., 0., 1., 1.], [
                                            0., 0., 0., 0.], [0., 0., 1., 1.]])

            B = np.random.randn(D, D, dout, din)
        else:
            A = np.random.randn(n, D, D, din)
            B = np.random.randn(n, D, D, dout, din)

        if self.trainable_A:
            self.register_parameter(name='A',
                                    param=nn.Parameter(torch.tensor(A, dtype=torch.float32).contiguous()))
        else:
            self.register_buffer(name='A', tensor=torch.tensor(
                A, dtype=torch.float32).contiguous())

        self.register_parameter(name='B',
                                param=nn.Parameter(torch.tensor(B, dtype=torch.float32).contiguous()))

        self.set_parallel_vars(n)

    def set_parallel_vars(self, n):
        m = int(np.log(n-1)/np.log(2.0))+1
        self.nbin = m

        self.positions = np.zeros([m, m], dtype=int)
        for i in range(0, m-1):
            k = 2
            for j in range(i+1, m):
                self.positions[i, j] = k
                k = 2*k

    def get_binary(self, i):
        return [int(c) for c in format(i, f'0{self.nbin}b')]

    def get_A(self, i):
        if self.ti:
            A = self.A
        else:
            A = self.A[i]
        return A

    def get_B(self, i):
        if self.ti:
            B = self.B
        else:
            B = self.B[i]
        return B

    def forward(self, input):
        n = input.shape[1]
        d = input.shape[2]

        eps_div = 1e-6

        if self.ti:
            # We can apply the model on inputs of different sizes
            # but we have to recalculate the positions and nbins for parallel execution
            if self.n != n:
                self.n = n
                self.set_parallel_vars(n)
        else:
            assert self.n == n, f"Input size is {n}. Expected an input tensor with {self.n} features."

        assert self.din == d, f"Input feature dimension is {d}. Expected an input features of size {self.din}."

        Alist = []
        for i in range(n):
            X = torch.reshape(input[:, i, :], [-1, self.din])
            A = self.get_A(i)
            # We use L2 normalized vectors for the TDVP
            X = torch.divide(X, torch.linalg.norm(X, axis=1, keepdims=True))
            A = torch.einsum("bi,lri->blr", X, A)
            Alist.append(A)

        M0 = torch.ones([1, 1])
        if self.mode == "parallel":
            if hasattr(self, "cyclic") and self.cyclic:
                raise Exception(
                    "The parallel forward pass for cyclic MPO is not yet implemented!")
            # Parallel computation of the local hamiltonian: complexity=log(n)d^3.
            # Left contractions
            Bleft = [Alist]
            for i in range(self.nbin-1):
                blist = []
                for i in range(len(Alist) // 2):
                    A1 = Alist[2 * i]
                    A2 = Alist[2 * i + 1]
                    blist.append(torch.einsum("aij,ajk->aik", A1, A2))
                Alist = blist
                Bleft.append(blist)
                if len(Bleft) == 1:
                    break

            Aleft = []
            for i in range(n):
                Al = None
                x = self.get_binary(i)
                xp = x@self.positions
                for j in range(self.nbin):
                    if x[j]:
                        A = Bleft[self.nbin-1-j][xp[j]]
                        if Al is None:
                            Al = A[:, 0, :]
                        else:
                            Al = torch.einsum("ai,aij->aj", Al, A)
                if Al is not None:
                    if hasattr(self, "scale"):
                        Al = Al/(torch.linalg.norm(Al, dim=-
                                                   1, keepdim=True)+eps_div)
                    Aleft.append(Al)
            # Right contractions
            Alist = Bleft[0]
            Bright = [Alist]
            for i in range(self.nbin-1):
                blist = []
                for i in range(len(Alist) // 2):
                    A1 = Alist[-2 * i - 2]
                    A2 = Alist[-2 * i - 1]
                    blist = [torch.einsum("aij,ajk->aik", A1, A2)] + blist
                Alist = blist
                Bright.append(blist)
                if len(Bright) == 1:
                    break

            Aright = []
            for i in range(n):
                # Ar = M0
                Ar = None
                x = self.get_binary(i)
                xp = x@self.positions
                for j in range(self.nbin):
                    if x[j]:
                        A = Bright[self.nbin-1-j][-1-xp[j]]
                        if Ar is None:
                            Ar = A[:, :, 0]
                        else:
                            Ar = torch.einsum("aij,aj->ai", A, Ar)
                if Ar is not None:
                    if hasattr(self, "scale"):
                        Ar = Ar/(torch.linalg.norm(Ar, dim=-
                                                   1, keepdim=True)+eps_div)
                    Aright = [Ar] + Aright
        else:
            if hasattr(self, "cyclic") and self.cyclic:
                # Sequentially calculating the left and the right Hamiltonians
                # Left part of the Hamiltonian
                Aleft = []
                for i in range(n - 1):
                    if i == 0:
                        A = Alist[i][:, :, :]
                    else:
                        A = torch.einsum("aij,ajk->aik", Aleft[-1], Alist[i])
                    A = A / \
                        (torch.linalg.norm(
                            A, dim=[-2, -1], keepdim=True)+eps_div)
                    Aleft.append(A)

                # Right part of the Hamiltonian
                Aright = []
                for i in range(n - 1):
                    if i == 0:
                        A = Alist[n-1-i][:, :, :]
                    else:
                        A = torch.einsum(
                            "aij,ajk->aik", Alist[n - 1 - i], Aright[0])
                    A = A / \
                        (torch.linalg.norm(
                            A, dim=[-2, -1], keepdim=True)+eps_div)
                    Aright = [A] + Aright
            else:
                Aleft = []
                for i in range(n - 1):
                    if i == 0:
                        A = Alist[i][:, 0, :]
                    else:
                        A = torch.einsum("ai,aij->aj", Aleft[-1], Alist[i])
                    A = torch.nn.functional.normalize(
                        A, dim=-1, eps=eps_div)
                    Aleft.append(A)

                # Right part of the Hamiltonian
                Aright = []
                for i in range(n - 1):
                    if i == 0:
                        A = Alist[n-1-i][:, :, 0]
                    else:
                        A = torch.einsum(
                            "aij,aj->ai", Alist[n - 1 - i], Aright[0])
                    A = torch.nn.functional.normalize(
                        A, dim=-1, eps=eps_div)
                    Aright = [A] + Aright

        # Calculate the local map
        Mlist = []
        for i in range(n):
            B = self.get_B(i)
            if hasattr(self, "cyclic") and self.cyclic:
                if i == 0:
                    H = torch.einsum(
                        "tljk,alt->ajk", B, Aright[i])
                elif i == n-1:
                    H = torch.einsum(
                        "ati,itjk->ajk", Aleft[i-1], B)
                else:
                    H = torch.einsum("ati,iljk,alt->ajk",
                                     Aleft[i-1], B, Aright[i])
            else:
                if i == 0:
                    H = torch.einsum(
                        "ljk,al->ajk", B[0, :, :, :], Aright[i])
                elif i == n-1:
                    H = torch.einsum(
                        "ai,ijk->ajk", Aleft[i-1], B[:, 0, :, :])
                else:
                    H = torch.einsum("ai,iljk,al->ajk",
                                     Aleft[i-1], B, Aright[i])

            nrm = torch.linalg.norm(H, dim=[-2, -1], keepdim=True)+eps_div
            H = torch.divide(H, nrm)
            H = self.scale * H

            if torch.isnan(H).any():
                H[torch.isnan(H)] = 0

            if self.activation == "expm":
                assert self.din == self.dout, "The input and output feature dimensions should be the same for the exponential activation"
                if self.din == 2:
                    b = (H[:, 0, 1]-H[:, 1, 0])/2.0
                    # This explicit calculation should improve the calculation of the matrix exponential...
                    H = torch.reshape(torch.stack([torch.cos(b), torch.sin(
                        b), -torch.sin(b), torch.cos(b)], axis=-1), [-1, 2, 2])
                else:
                    # For larger system sizes we have to calculate the exponential numerically
                    H = torch.matrix_exp((H-torch.transpose(H, 0, 1))/2.)
            Mlist.append(H)

        # Calculating the new mps
        mps = []
        for i in range(n):
            A = torch.einsum(
                "aij,aj->ai",
                Mlist[i],
                torch.reshape(input[:, i, :], [-1, self.din]),
            )
            if self.activation == "relu":
                # The slope below the threshold should not be 0 in order
                # to avoid dividing by 0 in the next step if normalization is used
                A = self.LeakyReLU(A)
            if self.activation == "sigmoid":
                A = self.sigmoid(A)
            if self.activation == "exp":
                A = torch.exp(A)
            mps.append(torch.reshape(A, [-1, self.dout]))

        x = torch.stack(mps, dim=1)
        if hasattr(self, "residual") and self.residual:
            x = (x + self.res_coef*input)/2.0

        # Final normalization of the vector
        if self.out_norm == "L2":
            x = torch.divide(x, (torch.linalg.norm(
                x, dim=-1, keepdim=True)+eps_div))
        elif self.out_norm == "L1":
            # We use also absolute value in order to avoid dividing by 0
            x = torch.abs(x)
            x = torch.divide(x, torch.sum(x, dim=-1, keepdim=True))

        return x, Aleft, Aright