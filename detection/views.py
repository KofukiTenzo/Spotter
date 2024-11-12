from django.shortcuts import render
from .services import Services
from django.core.files.storage import FileSystemStorage
from video_detector import settings

def index(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        
        output_video_url = Services().yolov8_ensemble_detection_on_video_with_consensus_voting(video_file)

        return render(request, 'detection\\index.html', {
            'output_video_url': output_video_url,
        })

    return render(request, 'detection\\index.html')