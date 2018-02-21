import threading
import time


class RenderThread(threading.Thread):
    def __init__(self, sess, trainer, environment, brain_name, normalize, fps):
        threading.Thread.__init__(self)
        self.sess = sess
        self.env = environment
        self.trainer = trainer
        self.brain_name = brain_name
        self.normalize = normalize
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.fps = fps

    def run(self):
        with self.sess.as_default():
            while True:
                with self.pause_cond:
                    done = False
                    info = self.env.reset()[self.brain_name]
                    while not done:
                        while self.paused:
                            self.pause_cond.wait()
                        t_s = time.time()
                        info = self.trainer.take_action(info, self.env, self.brain_name, 0, self.normalize,
                                                        stochastic=False)
                        done = info.local_done[0]
                        time.sleep(max(0, 1 / self.fps - (time.time() - t_s)))
                time.sleep(0.1)

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()
