#ifdef DFTFE_MINIMAL_COMPILE
template class kohnShamDFTOperatorDeviceClass<2,
                                              2,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<3,
                                              3,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              4,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              5,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              6,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              9,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;

template class kohnShamDFTOperatorDeviceClass<
  2,
  2,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  3,
  3,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  4,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  5,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  6,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  9,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
#else
template class kohnShamDFTOperatorDeviceClass<1,
                                              1,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<1,
                                              2,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<2,
                                              2,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<2,
                                              3,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<2,
                                              4,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<3,
                                              3,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<3,
                                              4,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<3,
                                              5,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<3,
                                              6,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              4,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              5,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              6,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<4,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              5,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              6,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              9,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<5,
                                              10,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              6,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              9,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              10,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              11,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<6,
                                              12,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              7,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              9,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              10,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              11,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              12,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              13,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<7,
                                              14,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              8,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              9,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              10,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              11,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              12,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              13,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              14,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              15,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<8,
                                              16,
                                              dftfe::utils::MemorySpace::HOST>;
template class kohnShamDFTOperatorDeviceClass<
  1,
  1,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  1,
  2,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  2,
  2,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  2,
  3,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  2,
  4,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  3,
  3,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  3,
  4,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  3,
  5,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  3,
  6,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  4,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  5,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  6,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  4,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  5,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  6,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  9,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  5,
  10,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  6,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  9,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  10,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  11,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  6,
  12,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  7,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  9,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  10,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  11,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  12,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  13,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  7,
  14,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  8,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  9,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  10,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  11,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  12,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  13,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  14,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  15,
  dftfe::utils::MemorySpace::DEVICE>;
template class kohnShamDFTOperatorDeviceClass<
  8,
  16,
  dftfe::utils::MemorySpace::DEVICE>;
#endif
