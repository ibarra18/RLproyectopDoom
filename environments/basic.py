import gym
from gym import spaces
from vizdoom import DoomGame, Mode, ScreenResolution

class BasicDoomEnv(gym.Env):
    def __init__(self):
        super(BasicDoomEnv, self).__init__()
        self.game = DoomGame()
        self.game.load_config("scenarios/basic.cfg")
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_window_visible(False)
        self.game.init()
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        if done:
            state = self.game.get_state()
            return state.screen_buffer, reward, done, {}
        else:
            return None, reward, done, {}

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        return state.screen_buffer

    def close(self):
        self.game.close()
