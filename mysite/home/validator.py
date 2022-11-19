
import math

class MyValidator:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_DOWN(self, mid_start):

        if len(self.tracks) >= 2:

            if self.state == '0':

                distance = math.sqrt(float((self.tracks[-1][1] - self.tracks[-2][1])**2) + float(
                    (self.tracks[-1][1] - self.tracks[-2][1])**2))
                if distance < 10:
                    if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                        state = '1'
                        self.dir = 'down'
                        return True
            else:
                return False
        else:
            return False

    def going_UP(self, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                distance = math.sqrt(float((self.tracks[-1][1] - self.tracks[-2][1])**2) + float(
                    (self.tracks[-1][1] - self.tracks[-2][1])**2))
                if distance < 10:
                    if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                        state = '1'
                        self.dir = 'up'
                        return True
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True