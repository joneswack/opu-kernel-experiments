

class EnergyMonitor():
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self._monitoring_period = 10 # in seconds
        self._nvidia_command = self._setup_command()


    @property
    def monitoring_period(self):
        return self._monitoring_period
    @setter.monitoring_period
    def monitoring_period(self,period):
        self._monitoring_period = period

    def _setup_command(self):
        com = "nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l "+str(self._monitoring_period)
        self._nvidia_command = com


    def start():
        print("Start monitoring GPU energy consumption.")



