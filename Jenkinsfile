pipeline{
    agent any

    environment{
        REGION = "ap-northeast-2"
        ECR_PATH = "595017219040.dkr.ecr.ap-northeast-2.amazonaws.com"
        ECR_IMAGE = "lyric-auto-tag-api-server"
        AWS_CREDENTIAL_ID = "aws-moon-credentials"
    }

    stages{
        stage("INITAL"){
            steps{
                echo "INITIALIZING..."
            }
        }

        stage("DOCKER BUILD"){
            steps{
                script{
                    docker.withRegistry("https://${ECR_PATH}","ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
                       image = docker.build("${ECR_PATH}/${ECR_IMAGE}","--network=host --no-cache .")}
                }
            }
        }

        // stage("TEST"){
        //     steps{
        //         script{
        //             docker.image("${ECR_PATH}/${ECR_IMAGE}").inside{
        //                 sh """python3 -m pytest tests"""}
        //         }
        //     }
        // }

        stage("IMAGE PUSH"){
            steps{
                script{
                    docker.withRegistry("https://${ECR_PATH}","ecr:${REGION}:${AWS_CREDENTIAL_ID}"){
                        image.push("v${env.BUILD_NUMBER}")}
                }
            }
        }
    }
    post{
        always{
            script{
                sh"""
                docker rmi ${ECR_PATH}/${ECR_IMAGE}:latest
                """
            }
        }
        success{
            script{
                sh"""
                docker rmi ${ECR_PATH}/${ECR_IMAGE}:v$BUILD_NUMBER
                """
            }
        }
    }
}
