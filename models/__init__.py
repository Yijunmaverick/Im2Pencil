def create_model(opt):
    model = None

    if opt.model == 'pix2pix3_twobranch':
        assert(opt.dataset_mode == 'aligned3')
        from .pix2pix_model3_twobranch import Pix2PixModel3_twobranch
        model = Pix2PixModel3_twobranch()
    elif opt.model == 'pix2pix2_twobranch':
        assert(opt.dataset_mode == 'aligned2')
        from .pix2pix_model2_twobranch import Pix2PixModel2_twobranch
        model = Pix2PixModel2_twobranch()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
