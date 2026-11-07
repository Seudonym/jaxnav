import numpy as np
import jax
import jax.numpy as jnp
import rclpy
import rclpy.time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tb3_mppi.mppi import mppi_step, H, DT

N_BEAMS = 180     


class JaxMPPI(Node):
    def __init__(self):
        super().__init__("jax_mppi")
        self.goal_msg = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose = None
        self.obstacles = None
        self.U = jnp.zeros((H, 2))
        self.key = jax.random.PRNGKey(0)

        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # warmpu
        mppi_step(self.key, jnp.zeros(3), self.U, jnp.zeros(2), jnp.zeros((N_BEAMS, 2)))
        self.get_logger().info("MPPI compiled")
        self.create_timer(DT, self.tick)

    def on_goal(self, msg):
        self.goal_msg = msg
        self.U = jnp.zeros((H, 2))  

    def goal_in_odom(self):
        p = self.goal_msg.pose.position
        frame = self.goal_msg.header.frame_id
        if frame == "odom":
            return np.array([p.x, p.y])
        try:
            t = self.tf_buffer.lookup_transform("odom", frame, rclpy.time.Time())
        except TransformException:
            self.get_logger().warn(f"Cannot transform goal from {frame} to odom")
            return None
        tr, q = t.transform.translation, t.transform.rotation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        c, s = np.cos(yaw), np.sin(yaw)        
        return np.array([tr.x + c * p.x - s * p.y, tr.y + s * p.x + c * p.y])


    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self.pose = np.array([p.x, p.y, yaw])

    def on_scan(self, msg):
        if self.pose is None:
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        idx = np.linspace(0, len(r) - 1, N_BEAMS).astype(int)       # downsample to fixed size
        r, ang = r[idx], ang[idx]
        bad = ~np.isfinite(r) | (r < msg.range_min) | (r > msg.range_max)
        r = np.where(bad, 10.0, r)                                  
        x, y, th = self.pose                                       
        self.obstacles = np.c_[x + r * np.cos(ang + th), y + r * np.sin(ang + th)]

    def tick(self):
        cmd = Twist()
        if self.pose is None or self.obstacles is None or self.goal_msg is None:
            self.cmd_pub.publish(cmd)
            return
        goal = self.goal_in_odom()
        if goal is None:
            self.cmd_pub.publish(cmd)
            return
        if np.linalg.norm(self.pose[:2] - np.asarray(goal)) > 0.1:
            self.key, u, self.U, _, _ = mppi_step(
                self.key, jnp.asarray(self.pose), self.U, goal, jnp.asarray(self.obstacles))
            cmd.linear.x, cmd.angular.z = float(u[0]), float(u[1])
        self.cmd_pub.publish(cmd)                                   # zero Twist = stop at goal


def main():
    rclpy.init()
    rclpy.spin(JaxMPPI())


if __name__ == "__main__":
    main()
