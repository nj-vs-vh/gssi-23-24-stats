import traceback
import corner  # type: ignore
import emcee  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # type: ignore


def run_analysis(idx: int, exp_time_yr: float):
    print(f"\n\nRunning analysis #{idx} for exposure = {exp_time_yr:.4f} years")
    M_DETECTOR_KG = 468
    M_100MO_KG = 250
    HALFTIME_2NUBB_YR = 7.12e18
    HALFTIME_0NUBB_YR = 5e23
    E_THRESHOLD_MEV = 30 / 1000

    Q_BETA_BETA_MEV = 3.034

    MC_BIN_SIZE = 5e-3
    MC_BIN_EDGES = np.arange(
        start=E_THRESHOLD_MEV,
        stop=Q_BETA_BETA_MEV + 0.1,
        step=MC_BIN_SIZE,
    )

    import dataclasses

    def bin_sizes_from_centers(bin_centers: np.ndarray) -> np.ndarray:
        assert len(bin_centers.shape) == 1
        deltas = bin_centers[1:] - bin_centers[:-1]
        sizes = np.zeros(shape=(bin_centers.size,))
        sizes[0] = deltas[0]
        sizes[-1] = deltas[-1]
        sizes[1:-1] = 0.5 * (deltas[1:] + deltas[:-1])
        return sizes

    # def normalize_to_pdf

    @dataclasses.dataclass
    class Spectrum:
        energy: np.ndarray
        pdf: np.ndarray

        def __post_init__(self) -> None:
            self.bin_sizes = bin_sizes_from_centers(self.energy)
            self.cdf = np.cumsum(self.pdf * self.bin_sizes)

        def generate_sample(self, size: int) -> np.ndarray:
            cdf_image = np.sort(np.random.uniform(low=0, high=1, size=size))
            return np.interp(cdf_image, self.cdf, self.energy)

        def rebinned(self, bin_edges: np.ndarray) -> "Spectrum":
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_edge_values = np.interp(
                bin_edges,
                self.energy,
                self.pdf,
                left=0,
                right=0,
            )
            # assuming linearity within one bin -- will work only with small enough bins, but ok for us
            bin_average_probabilities = 0.5 * (
                bin_edge_values[1:] + bin_edge_values[:-1]
            )
            return Spectrum(energy=bin_centers, pdf=bin_average_probabilities)

        def generate_histogram(
            self,
            total_count: float,
        ) -> np.ndarray:
            bin_average_counts = self.pdf * self.bin_sizes * total_count
            return np.random.poisson(lam=bin_average_counts)

        @classmethod
        def from_counts(cls, energy: np.ndarray, counts: np.ndarray) -> "Spectrum":
            probability = counts / (counts * bin_sizes_from_centers(energy)).sum()
            return Spectrum(energy=energy, pdf=probability)

        @classmethod
        def load(cls, filename: str) -> "Spectrum":
            raw = np.loadtxt(filename, skiprows=1)
            assert raw.shape[1] == 3
            energy = raw[:, 0] / 1000
            dnde = raw[:, 1]
            return Spectrum.from_counts(energy=energy, counts=dnde)

    @dataclasses.dataclass
    class DoubleElectronSpectrum:
        pdf: np.ndarray  # 2 dim
        x_energy: np.ndarray  # 1 dim
        y_energy: np.ndarray  # 1 dim

        def sum_spectrum(self) -> Spectrum:
            sum_binning_step = 5e-3

            e_min = np.inf
            e_max = 0.0
            it = np.nditer(self.pdf, flags=["multi_index"])
            for prob in it:
                idx_x, idx_y = it.multi_index
                if prob < 0:  # type: ignore
                    continue
                e = float(self.x_energy[idx_x] + self.y_energy[idx_y])
                e_min = min(e_min, e)
                e_max = max(e_max, e)

            assert np.isfinite(e_min)
            assert np.isfinite(e_max)
            assert e_min < e_max
            range_ = e_max - e_min
            e_bin_edges = np.arange(
                e_min - (range_ * 1e-3),
                e_max + sum_binning_step + (range_ * 1e-3),
                sum_binning_step,
            )
            e_bin_centers = 0.5 * (e_bin_edges[1:] + e_bin_edges[:-1])
            histogram = np.zeros_like(e_bin_centers)

            it = np.nditer(self.pdf, flags=["multi_index"])
            for prob in it:
                idx_x, idx_y = it.multi_index
                if prob < 0:  # type: ignore
                    continue
                e = self.x_energy[idx_x] + self.y_energy[idx_y]
                e_idx = np.digitize(e, e_bin_edges)
                histogram[e_idx] += prob

            return Spectrum.from_counts(energy=e_bin_centers, counts=histogram)

        def plot(self) -> None:
            ax: plt.Axes
            fig, ax = plt.subplots()

            img = ax.imshow(self.pdf, origin="lower", cmap="inferno")

            x_ticks = np.linspace(0, self.pdf.shape[0] - 1, 5, dtype="int")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                labels=["{:.2f}".format(self.x_energy[idx]) for idx in x_ticks]
            )
            ax.set_xlabel("$ \epsilon_1,  \\text{MeV} $")

            y_ticks = np.linspace(0, self.pdf.shape[1] - 1, 5, dtype="int")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(
                labels=["{:.2f}".format(self.y_energy[idx]) for idx in y_ticks]
            )
            ax.set_ylabel("$ \epsilon_2, \\text{MeV} $")

            ax.axvline(E_THRESHOLD_MEV, color="red", linestyle="--", label="Threshold")
            ax.axhline(E_THRESHOLD_MEV, color="red", linestyle="--")

            fig.colorbar(img, label="PDF")

        @classmethod
        def load(cls, filename: str) -> "DoubleElectronSpectrum":
            """
            Legend:
            - Column 1, 2 → Bin index for x and y directions.
            - Column 3, 4 → Bin center along x and y direction, corresponding to energy of first and second electron.
            - Column 5 → Probability content for the bin.
            """
            raw = np.loadtxt(filename)
            assert raw.shape[1] == 5
            x_min = int(raw[:, 0].min())
            x_max = int(raw[:, 0].max())
            y_min = int(raw[:, 1].min())
            y_max = int(raw[:, 1].max())

            probability_map = np.zeros(shape=(x_max - x_min + 1, y_max - y_min + 1))
            x_energy = np.zeros(shape=(x_max - x_min + 1))
            y_energy = np.zeros(shape=(y_max - y_min + 1))

            for x_idx, y_idx, x_e, y_e, prob in raw:
                x_idx = int(x_idx) - x_min
                y_idx = int(y_idx) - y_min
                probability_map[x_idx, y_idx] = prob
                x_energy[x_idx] = x_e
                y_energy[y_idx] = y_e

            # normalizing to integral 1
            probability_map /= (
                probability_map
                * np.tile(
                    bin_sizes_from_centers(x_energy).reshape((-1, 1)),
                    (1, y_max - y_min + 1),
                )
                * np.tile(
                    bin_sizes_from_centers(y_energy).reshape((1, -1)),
                    (x_max - x_min + 1, 1),
                )
            ).sum()

            return DoubleElectronSpectrum(probability_map, x_energy, y_energy)

    double_bb_2d = DoubleElectronSpectrum.load("data/100Mo_ssd_2ds.txt")
    # double_bb_2d.plot()
    double_bb = double_bb_2d.sum_spectrum().rebinned(MC_BIN_EDGES)

    majoron_2d = DoubleElectronSpectrum.load("data/m1_2ds.dat")
    # majoron_2d.plot()
    majoron = majoron_2d.sum_spectrum().rebinned(MC_BIN_EDGES)

    BETA_BACKGROUND_ACTIVITY_MEAN = 3.4e-4
    BETA_BACKGROUND_ACTIVITY_STD = 0.4e-4
    SEC_PER_YEAR = 365 * 86400

    sr_beta = Spectrum.load("data/90Sr-beta-decay.txt").rebinned(MC_BIN_EDGES)
    y_beta = Spectrum.load("data/90Y-beta-decay.txt").rebinned(MC_BIN_EDGES)

    SR_BETA_PLOT_KW = dict(label="$^{90} \\text{Sr}$ $ \\beta $-decay", color="magenta")
    Y_BETA_PLOT_KW = dict(label="$^{90} \\text{Y}$ $ \\beta $-decay", color="green")
    DOUBLE_BB_PLOT_KW = dict(label="$2\\nu \\beta \\beta$", color="blue")
    MAJORON_PLOT_KW = dict(label="$0\\nu \\beta \\beta$", color="orange")

    # fig, ax = plt.subplots()

    # ax.plot(sr_beta.energy, sr_beta.pdf, **SR_BETA_PLOT_KW)
    # ax.plot(y_beta.energy, y_beta.pdf, **Y_BETA_PLOT_KW)
    # ax.plot(double_bb.energy, double_bb.pdf, **DOUBLE_BB_PLOT_KW)
    # ax.plot(majoron.energy, majoron.pdf, **MAJORON_PLOT_KW)

    # ax.axvline(E_THRESHOLD_MEV, color="black", linestyle="--", label="Threshold")
    # ax.axvline(
    #     Q_BETA_BETA_MEV, color="red", linestyle="--", label="$Q_{\\beta \\beta}$"
    # )

    # ax.set_xlabel("E, MeV")
    # ax.set_ylabel("PDF")
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0, right=3.2)
    # ax.legend()
    # plt.show()

    N_A = 6.022e23
    MOLAR_MASS_100M_KG_MOL = 100 / 1000
    N0_100Mo = N_A * M_100MO_KG / MOLAR_MASS_100M_KG_MOL

    N_double_bb: float = N0_100Mo * np.log(2) * exp_time_yr / HALFTIME_2NUBB_YR
    N_majoron: float = N0_100Mo * np.log(2) * exp_time_yr / HALFTIME_0NUBB_YR

    def generate_beta_backgrounds() -> tuple[float, float]:
        return tuple(  # type: ignore
            np.random.normal(
                loc=BETA_BACKGROUND_ACTIVITY_MEAN,
                scale=BETA_BACKGROUND_ACTIVITY_STD,
                size=2,
            )
            * M_DETECTOR_KG
            * exp_time_yr
            * SEC_PER_YEAR
        )

    print("Signals:")
    print(f"{N_double_bb = :.2e}")
    print(f"{N_majoron = :.2e}")
    print()

    N_sr_y_beta_mean = (
        BETA_BACKGROUND_ACTIVITY_MEAN * M_DETECTOR_KG * exp_time_yr * SEC_PER_YEAR
    )
    N_sr_y_beta_std = (
        BETA_BACKGROUND_ACTIVITY_STD * M_DETECTOR_KG * exp_time_yr * SEC_PER_YEAR
    )
    print(f"Background: {N_sr_y_beta_mean:.2e} +/- {N_sr_y_beta_std:.2e}")
    print()
    print("Example background realization:")
    N_sr_beta_rlz, N_y_beta_rlz = generate_beta_backgrounds()
    print(f"{N_sr_beta_rlz = :.2e}")
    print(f"{N_y_beta_rlz = :.2e}")

    # fig, ax = plt.subplots()

    # N_sr_beta_rlz, N_y_beta_rlz = generate_beta_backgrounds()
    # ax.plot(sr_beta.energy, N_sr_beta_rlz * sr_beta.pdf, **SR_BETA_PLOT_KW)
    # ax.plot(y_beta.energy, N_y_beta_rlz * y_beta.pdf, **Y_BETA_PLOT_KW)
    # ax.plot(double_bb.energy, N_double_bb * double_bb.pdf, **DOUBLE_BB_PLOT_KW)
    # ax.plot(majoron.energy, N_majoron * majoron.pdf, **MAJORON_PLOT_KW)

    # ax.axvline(E_THRESHOLD_MEV, color="black", linestyle="--", label="Threshold")
    # ax.axvline(
    #     Q_BETA_BETA_MEV, color="red", linestyle="--", label="$Q_{\\beta \\beta}$"
    # )

    # ax.set_xlabel("E, MeV")
    # ax.set_ylabel("Counts per MeV")
    # ax.set_yscale("log")
    # ax.set_ylim(5)
    # ax.set_xlim(left=0, right=3.2)
    # ax.legend()
    # plt.show()

    def generate_data() -> tuple[np.ndarray, float, float]:
        N_sr_beta_rlz, N_y_beta_rlz = generate_beta_backgrounds()

        histogram = np.zeros(shape=(MC_BIN_EDGES.size - 1))
        for N_events, spectrum in zip(
            [N_double_bb, N_majoron, N_sr_beta_rlz, N_y_beta_rlz],
            [double_bb, majoron, sr_beta, y_beta],
        ):
            histogram += spectrum.generate_histogram(total_count=N_events)

        return histogram, N_sr_beta_rlz, N_y_beta_rlz

    histogram, N_sr_beta_rlz, N_y_beta_rlz = generate_data()

    # fig, ax = plt.subplots()
    # ax: plt.Axes

    # ax.stairs(
    #     histogram / MC_BIN_SIZE,  # counts -> counts per mev conversion
    #     MC_BIN_EDGES,
    #     label="Monte Carlo",
    #     linewidth=2,
    #     color="black",
    # )

    # ax.plot(sr_beta.energy, N_sr_beta_rlz * sr_beta.pdf, **SR_BETA_PLOT_KW)
    # ax.plot(y_beta.energy, N_y_beta_rlz * y_beta.pdf, **Y_BETA_PLOT_KW)
    # ax.plot(double_bb.energy, N_double_bb * double_bb.pdf, **DOUBLE_BB_PLOT_KW)
    # ax.plot(majoron.energy, N_majoron * majoron.pdf, **MAJORON_PLOT_KW)

    # ax.legend()

    # ax.set_xlabel("E, MeV")
    # ax.set_ylabel("Counts per MeV")
    # ax.set_yscale("log")
    # ax.set_xlim(MC_BIN_EDGES[0], MC_BIN_EDGES[-1])
    # ax.legend()
    # plt.show()

    def log_prior_1(theta: np.ndarray):
        N_double_bb, N_majoron, N_sr_beta, N_y_beta = theta
        return (
            stats.uniform.logpdf(N_double_bb, loc=0, scale=1e11)
            + stats.uniform.logpdf(N_majoron, loc=0, scale=1e11)
            + stats.norm.logpdf(N_sr_beta, loc=N_sr_y_beta_mean, scale=N_sr_y_beta_std)
            + stats.norm.logpdf(N_y_beta, loc=N_sr_y_beta_mean, scale=N_sr_y_beta_std)
        )

    def log_posterior_1(theta: np.ndarray, histogram: np.ndarray):
        logpi = log_prior_1(theta)
        if not np.isfinite(logpi):
            # print(f"prior is not finite for {theta = }")
            # raise RuntimeError()
            pass
        if np.isfinite(logpi):
            lambdas = np.zeros_like(histogram)
            for N_component, spectrum in zip(
                theta, [double_bb, majoron, sr_beta, y_beta]
            ):
                lambdas += N_component * spectrum.pdf * spectrum.bin_sizes
            loglike = stats.poisson.logpmf(histogram, np.clip(lambdas, 0, None)).sum()
            if not np.isfinite(loglike):
                # print(f"likelihood is not finitie for {theta = }")
                # raise RuntimeError()
                pass
            return logpi + loglike
        else:
            return logpi

    histogram, N_sr_beta_rlz, N_y_beta_rlz = generate_data()

    n_walkers = 64
    n_dim = 4
    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=n_dim,
        log_prob_fn=log_posterior_1,
        args=[histogram],
    )

    total_events = np.sum(histogram)
    # assuming that most events come from 2-neutrino beta decay
    initial_theta_mean = np.array(
        [
            total_events,
            total_events * 1e-4,
            N_sr_y_beta_mean,
            N_sr_y_beta_mean,
        ]
    )
    initial_theta_std = np.array(
        [
            total_events * 0.1,
            total_events * 1e-5,
            N_sr_y_beta_std,
            N_sr_y_beta_std,
        ]
    )
    initial_state = stats.norm.rvs(
        loc=initial_theta_mean, scale=initial_theta_std, size=(n_walkers, 4)
    )

    print("Sampling...")
    sampler.run_mcmc(initial_state, nsteps=7000, progress=True)
    print(f"{sampler.acceptance_fraction.mean() = }")
    tau = sampler.get_autocorr_time()
    print(f"{tau = }")
    burn_in = 3 * int(tau.max())
    thin = int(tau.max())
    print(f"{burn_in = } {thin = }")
    sample: np.ndarray = sampler.get_chain(flat=True, discard=burn_in, thin=thin)
    print(f"Sample ready, shape: {sample.shape}")

    fig: plt.Figure = corner.corner(
        sample,
        labels=["$N_{2\\nu\\beta\\beta}$", "$N_{Maj}$", "$N_{Sr}$", "$N_{Y}$"],
        truths=[N_double_bb, N_majoron, N_sr_beta_rlz, N_y_beta_rlz],
        truth_color="red",
        show_titles=True,
        title_fmt=".2e",
        quantiles=[0.05, 0.5, 0.95],
    )

    if exp_time_yr >= 1:
        exp_time_str = f"{exp_time_yr:.1f} years"
    elif exp_time_yr > 30 / 365:
        exp_time_str = f"{exp_time_yr * 365 / 30:.1f} months"
    else:
        exp_time_str = f"{exp_time_yr * 365:.1f} days"

    # fig.suptitle(f"Analysis results, exposure {exp_time_str}", fontsize=14)
    fig.savefig(f"out/analysis-pt1-{idx}-{exp_time_str.replace(' ', '-')}.pdf")
    fig.savefig(f"out/analysis-pt1-{idx}-{exp_time_str.replace(' ', '-')}.png")
    print("Done!")


if __name__ == "__main__":
    for idx, exp_time_yr in enumerate(
        [
            1 / 365,
            7 / 365,
            30 / 365,
            6 * 30 / 365,
            1,
            5,
            10,
            20,
        ]
    ):
        try:
            run_analysis(idx, exp_time_yr)
        except Exception:
            print("Error running analysis")
            traceback.print_exc()
