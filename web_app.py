import redis
from flask import Flask, render_template,request, Response


class WebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.redis_client = redis.Redis('localhost', port=6379)

        # Routing
        self.app.add_url_rule('/', view_func=self.index)
        self.app.add_url_rule('/start', view_func=self.start_program, methods=['POST'])
        self.app.add_url_rule('/stop', view_func=self.stop_program, methods=['POST'])
        self.app.add_url_rule('/video_feed', view_func=self.video_feed)


    def index(self):
        return render_template('index.html')

    def start_program(self):
        self.redis_client.set('command', 'start')
        return '', 204

    def stop_program(self):
        self.redis_client.set('command', 'stop')
        return '', 204

    def video_feed(self):
        def generate():
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe('video_stream')

            for message in pubsub.listen():
                if message['type'] == 'message':
                    frame = message['data']
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self):
        self.app.run(debug=True)





if __name__ == '__main__':
    web_app = WebApp()
    web_app.run()
