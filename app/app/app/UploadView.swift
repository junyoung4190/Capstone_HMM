import PhotosUI
import SwiftUI

struct UploadView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var goResult = false
    
    var body: some View {
        VStack {
            Text("딥페이크 노이즈")
            
            PhotosPicker(selection: $selectedItem, matching: .images) {
                Text("사진 선택하기")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
            }
            
            NavigationLink("", isActive: $goResult) {
                ResultLoadingView(image: selectedImage!)
            }
        }
        .onChange(of: selectedItem) { newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    selectedImage = uiImage
                    goResult = true   // 👉 자동 이동
                }
            }
        }
    }
}
