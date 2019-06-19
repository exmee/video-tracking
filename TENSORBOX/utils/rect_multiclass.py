class Rect_Multiclass(object):
    def __init__(self, cx, cy, width, height, confidence, label_confidence,label):
        
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.true_confidence = confidence
        self.label_confidence = label_confidence
        self.label= label
        self.trackID=-1
        self.x1 = self.cx - self.width/2.
        self.x2 = self.cx + self.width/2.
        self.y1 = self.cy - self.height/2.
        self.y2 = self.cy + self.height/2.

    def overlaps(self, other):
        if abs(self.cx - other.cx) > (self.width + other.width) / 1.5:
            return False
        elif abs(self.cy - other.cy) > (self.height + other.height) / 2.0:
            return False
        else:
            return True
    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))
    def intersection(self, other):
        left = max(self.cx - self.width/2., other.cx - other.width/2.)
        right = min(self.cx + self.width/2., other.cx + other.width/2.)
        width = max(right - left, 0)
        top = max(self.cy - self.height/2., other.cy - other.height/2.)
        bottom = min(self.cy + self.height/2., other.cy + other.height/2.)
        height = max(bottom - top, 0)
        return width * height
    def area(self):
        return self.height * self.width
    def union(self, other):
        return self.area() + other.area() - self.intersection(other)
    def iou(self, other):
        return self.intersection(other) / self.union(other)
    def __eq__(self, other):
        return (self.cx == other.cx and 
            self.cy == other.cy and
            self.width == other.width and
            self.height == other.height and
            self.confidence == other.confidence and
            self.label_confidence == other.label_confidence and self.label == other.label and self.trackID == other.trackID)
