import matplotlib.pyplot as plt
import os

# define if saves or show
class PlotFromCLI(object):

    @staticmethod
    def add_arguments_to_parser(parser):
        parser.add_argument('--plot_style', nargs='*', default=[],
                            help='plot styles to be used')
        parser.add_argument('--ext', type=str, default='png',
                            help='extension used to save the figres.')
        parser.add_argument('-s', '--save', type=str,
                            help='save directory, otherwise just show.')
        return parser

    def __init__(self, **kwargs):
        self.save = kwargs['save']
        self.ext = kwargs['ext']
        if self.save is not None:
            if not os.path.isdir(self.save):
                os.mkdir(self.save)
        if kwargs['plot_style']:
            plt.style.use(kwargs['plot_style'])

    def show(self, title):
        if self.save is not None:
            if self.save == 'dontsave':
                pass
            else:
                plt.savefig(os.path.join(self.save, title.lower().replace(' ', '_') + '.' + self.ext))
        else:
            plt.tight_layout()
            plt.title(title)
            plt.show()
