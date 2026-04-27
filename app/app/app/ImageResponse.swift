import UIKit

struct ImageResponse: Codable {
    let imageId: Int
    let originalUrl: String
    let processedUrl: String
    let riskScore: Double
}

func uploadImage(image: UIImage, completion: @escaping (ImageResponse?) -> Void) {
    guard let url = URL(string: "http://127.0.0.1:8080/api/images/upload") else { return }
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    var data = Data()
    
    let imageData = image.jpegData(compressionQuality: 0.8)!
    
    data.append("--\(boundary)\r\n".data(using: .utf8)!)
    data.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
    data.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    data.append(imageData)
    data.append("\r\n".data(using: .utf8)!)
    
    data.append("--\(boundary)\r\n".data(using: .utf8)!)
    data.append("Content-Disposition: form-data; name=\"option\"\r\n\r\n".data(using: .utf8)!)
    data.append("strong\r\n".data(using: .utf8)!)
    
    data.append("--\(boundary)--\r\n".data(using: .utf8)!)
    
    request.httpBody = data
    
    URLSession.shared.dataTask(with: request) { data, _, _ in
        guard let data = data else { return }
        
        let result = try? JSONDecoder().decode(ImageResponse.self, from: data)
        
        DispatchQueue.main.async {
            completion(result)
        }
    }.resume()
}
