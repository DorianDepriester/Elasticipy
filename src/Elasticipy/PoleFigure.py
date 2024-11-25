import numpy as np
from matplotlib import transforms as mtransforms
from matplotlib import scale as mscale
from matplotlib.ticker import FixedLocator
from matplotlib.projections.polar import ThetaFormatter

def format_coord(phi, theta):
    """
    Format the coordinates as follows: 'phi=val, theta=val', wheres vals are in degrees
    :param phi: azimuth
    :param theta: colatitude
    """
    return ('azimuth=%0.1f\N{DEGREE SIGN}, '
            'colatitude=%0.1f\N{DEGREE SIGN}'
            % (np.degrees(phi) % 360, np.degrees(theta)))
    
class StereographicScale(mscale.ScaleBase):
    name = 'stereographic'

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator(np.radians(np.arange(0, 90, 20))))
        axis.set_major_formatter(ThetaFormatter())  # Use the same formatter as for phi (called theta in polar)
        axis.set_minor_formatter(ThetaFormatter())

    def get_transform(self):
        return self.StereographicTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return 0, min(vmax, np.pi / 2)

    class StereographicTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, theta):
            return np.tan(theta / 2)

        def inverted(self):
            return StereographicScale.InvertedStereographicTransform()

    class InvertedStereographicTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, r):
            return 2 * np.arctan(r)

        def inverted(self):
            return StereographicScale.StereographicTransform()


mscale.register_scale(StereographicScale)


class LambertScale(mscale.ScaleBase):
    name = 'lambert'

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator(np.radians(np.arange(0, 90, 15))))
        axis.set_major_formatter(ThetaFormatter())  # Use the same formatter as for phi (called theta in polar)
        axis.set_minor_formatter(ThetaFormatter())

    def get_transform(self):
        return self.LambertTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return 0, min(vmax, np.pi / 2)

    class LambertTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, theta):
            return 2 * np.sin(theta / 2)

        def inverted(self):
            return LambertScale.InvertedLambertTransform()

    class InvertedLambertTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, r):
            theta = np.zeros(len(r))
            theta[r >= np.sqrt(2)] = np.pi / 2
            theta[r < np.sqrt(2)] = 2 * np.arcsin(r[r < np.sqrt(2)] / 2)
            return theta

        def inverted(self):
            return LambertScale.LambertTransform()


mscale.register_scale(LambertScale)

def add_polefigure(fig, *args, projection='stereographic', **kwargs):
    if projection.lower() == 'equal area':
        projection = 'lambert'
    ax = fig.add_subplot(*args, projection='polar', **kwargs)
    ax.set_rlim(0.0, np.pi / 2)
    ax.set_ylim(0.0, 2*np.pi)
    ax.set_rscale(projection)
    ax.format_coord = format_coord
    return ax