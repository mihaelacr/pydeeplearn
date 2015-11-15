def ImportMatplotlibPlot():
  import matplotlib
  import os
  havedisplay = "DISPLAY" in os.environ
  if not havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    havedisplay = (exitval == 0)
  if havedisplay:
    import matplotlib.pyplot as plt
  else:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt