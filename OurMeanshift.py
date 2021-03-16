import numpy as np
import cv2

def our_meanShift(_probImage, window, criteria ):
    #CV_INSTRUMENT_REGION()

    #size
    #cn
    #mat
    #umat
    isUMat = _probImage.isUMat()

    if (isUMat):
        umat = _probImage.getUMat(), cn = umat.channels(), size = umat.size()
    else:
        mat = _probImage.getMat(), cn = mat.channels(), size = mat.size()

    cur_rect = window

    CV_Assert( cn == 1 )

    if window.height <= 0 or window.width <= 0:
        print("Input window has non-positive sizes" )

    window = window & Rect(0, 0, size.width, size.height)

    eps = (criteria.type & TermCriteria::EPS) ? std::max(criteria.epsilon, 0.) : 1.;
    eps = cvRound(eps*eps);
    int i, niters = (criteria.type & TermCriteria::MAX_ITER) ? std::max(criteria.maxCount, 1) : 100;

    for( i = 0; i < niters; i++ ):
        cur_rect = cur_rect & Rect(0, 0, size.width, size.height);
        if( cur_rect == Rect() )
        {
            cur_rect.x = size.width/2;
            cur_rect.y = size.height/2;
        }
        cur_rect.width = std::max(cur_rect.width, 1);
        cur_rect.height = std::max(cur_rect.height, 1);

        Moments m = isUMat ? moments(umat(cur_rect)) : moments(mat(cur_rect));

        # Calculating center of mass
        if( fabs(m.m00) < DBL_EPSILON )
            break;

        int dx = cvRound( m.m10/m.m00 - window.width*0.5 );
        int dy = cvRound( m.m01/m.m00 - window.height*0.5 );

        int nx = std::min(std::max(cur_rect.x + dx, 0), size.width - cur_rect.width);
        int ny = std::min(std::max(cur_rect.y + dy, 0), size.height - cur_rect.height);

        dx = nx - cur_rect.x;
        dy = ny - cur_rect.y;
        cur_rect.x = nx;
        cur_rect.y = ny;

        # Check for coverage centers mass & window
        if( dx*dx + dy*dy < eps )
            break;
    }

    window = cur_rect;
    return i;