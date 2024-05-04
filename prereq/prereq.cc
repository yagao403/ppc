struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    double c1 = 0;
    double c2 = 0;
    double c3 = 0;

    for (int x = x0; x<x1; x++){
        for (int y = y0; y<y1; y++){
            int index = 3 * x + 3 * nx * y;
            c1 += data[0 + index];
            c2 += data[1 + index];
            c3 += data[2 + index];
        }
    }
    int n = (y1 - y0) * (x1 - x0);
    Result result{{static_cast<float>(c1/n), static_cast<float>(c2/n), static_cast<float>(c3/n)}};
    
    return result;
}
