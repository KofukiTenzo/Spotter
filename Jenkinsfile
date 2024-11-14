pipeline {
    
    stages {
        stage("Clone code") {
            steps {
                echo "Clone code from the GitHub repository"
                git url: "https://github.com/KofukiTenzo/spotter.git", branch: "main"
            }
        }
        
        stage("Build") {
            steps {
                echo "Building Image"
                sh "docker build -t spotter ."
            }
        }
        
        stage("Push to DockerHub") {
            steps {
                echo "Push build image to DockerHub"
        
                withCredentials([usernamePassword(credentialsId: "DockerHubAcc", passwordVariable: "DockerHubPass", usernameVariable: "DockerHubUser")]){
                    sh '''
                    docker login -u $DockerHubUser -p $DockerHubPass
                    docker tag spotter:latest $DockerHubUser/spotter:latest
                    docker push $DockerHubUser/spotter:latest
                    '''
                }
            }
        }
        
        stage('Update Docker Image') {
            steps {
                echo 'Updating passwc image on latest'
                sh 'docker pull kofuki/spotter:latest'
            }
        }

        stage('Stop and remove old container') {
            steps {
                echo 'Stoping old spotter container'
                sh 'docker stop spotter'

                echo 'Removing old spotter container'
                sh 'docker rm spotter'
            }
        }
      
    }
}
