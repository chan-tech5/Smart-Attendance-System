from apscheduler.schedulers.background import BackgroundScheduler


class AttendanceScheduler:
    def __init__(self, storage):
        self.storage = storage
        self.scheduler = BackgroundScheduler()
        self.job = None
        self.cutoff_hour = 9
        self.cutoff_minute = 0

    def start(self):
        self.scheduler.start()
        self._apply_schedule()

    def set_cutoff(self, hour, minute):
        self.cutoff_hour = hour
        self.cutoff_minute = minute
        self._apply_schedule()

    def _apply_schedule(self):
        if self.job:
            try:
                self.job.remove()
            except Exception:
                pass
        self.job = self.scheduler.add_job(
            self.storage.mark_absentees,
            'cron',
            hour=self.cutoff_hour,
            minute=self.cutoff_minute,
            misfire_grace_time=300
        )
        print(f"Cutoff scheduled for {self.cutoff_hour:02}:{self.cutoff_minute:02}")

    def shutdown(self):
        self.scheduler.shutdown(wait=False)
