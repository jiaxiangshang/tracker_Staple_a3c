function out = getFeatureMap(im_patch, feature_type, cf_response_size, hog_cell_size)

% code from DSST

% allocate space
switch feature_type
    case 'fhog'
        %im_patch(1:10,1:10,1)
        temp = fhog(single(im_patch), hog_cell_size);
        %temp(1:10,1:10,1)
        h = cf_response_size(1);
        w = cf_response_size(2);
        out = zeros(h, w, 28, 'single');
        out(:,:,2:28) = temp(:,:,1:27);
        if hog_cell_size > 1
            im_patch = mexResize(im_patch, [h, w] ,'auto');
            %im_patch(1:10,1:10,1)
        end
        % if color image
        if size(im_patch, 3) > 1
            im_patch = rgb2gray(im_patch);
        end
        out(:,:,1) = single(im_patch)/255 - 0.5;
        %out(1:10,1:10,1)
    case 'gray'
        if hog_cell_size > 1, im_patch = mexResize(im_patch,cf_response_size,'auto');   end
        if size(im_patch, 3) == 1
            out = single(im_patch)/255 - 0.5;
        else
            out = single(rgb2gray(im_patch))/255 - 0.5;
        end        
end
        
end

