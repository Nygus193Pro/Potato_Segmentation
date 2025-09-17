from test_tracker import MainProgram

program = MainProgram(
    model_path="assets/models/best.xml",
    video_path="assets/videos/demo.mp4",
    categorizer_threshold=10000,
    use_redis=False
)

program.start_processing()
